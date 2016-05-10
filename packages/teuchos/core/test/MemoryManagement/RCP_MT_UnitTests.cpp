/*
// @HEADER
// ***********************************************************************
//
//                    Teuchos: Common Tools Package
//                 Copyright (2004) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ***********************************************************************
// @HEADER
*/

// #define KNOCK_OUT_TEMP

#include "General_MT_UnitTests.hpp"

#ifdef HAVE_TEUCHOSCORE_CXX11

#include "Teuchos_RCP.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include <vector>
#include <thread>

//#define KNOCK_OUT_ALL_CODE

namespace {
#ifndef KNOCK_OUT_ALL_CODE

using Teuchos::null;
using Teuchos::RCP;
using Teuchos::rcp;

static std::atomic<bool> s_bAllowThreadsToRun;				// this static is used to spin lock all the threads - after we allocate we set this true and all threads can complete

//
// Unit Test 2: mtCreateIndependentRCP
// Test debug node tracing thread safety
// Restore failure by defining BREAK_MUTEX_WHICH_PROTECTS_DEBUG_NODE_TRACING at the top (and run debug mode) which removes protective mutex on RCPNodeTracer::removeRCPNode() and RCPNodeTracer::addNewRCPNode()
//
static void create_independent_rcp_objects(int numAllocations) {
  for(int n = 0; n < numAllocations; ++n ) {
    RCP<int> ptr( new int ); // this allocates a new rcp ptr independent of all other rcp ptrs, and then dumps it, over and over
  }
}

TEUCHOS_UNIT_TEST( RCP, mtCreateIndependentRCP )
{
  const int numThreads = 16;
  const int numRCPAllocations = 10000;
  try {
    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i) {
      threads.push_back(std::thread(create_independent_rcp_objects, numRCPAllocations));
    }
    for (int i = 0; i < threads.size(); ++i) {
      threads[i].join();
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
}

TEUCHOS_UNIT_TEST( RCP, mtRCPWeakToStrongSimple )
{
  RCP<CatchMemoryLeak> strongPtr(new CatchMemoryLeak);
  RCP<CatchMemoryLeak> weakPtr = strongPtr.create_weak();
  strongPtr = null;
  RCP<CatchMemoryLeak> weakPtrFromWeak = weakPtr.create_weak();  // this is ok - we have a weakPtr and make a copy
  TEST_EQUALITY_CONST(weakPtrFromWeak.weak_count(), 2);
  TEST_EQUALITY_CONST(weakPtrFromWeak.strong_count(), 0);
  RCP<CatchMemoryLeak> strongPtrFromWeak = weakPtr.create_strong();  // this should return null - we have a weakPtr but no strong count so cannot make a new strong
  TEST_EQUALITY_CONST(strongPtrFromWeak.weak_count(), 0);
  TEST_EQUALITY_CONST(strongPtrFromWeak.strong_count(), 0);

}

// used to track conversions by the threads from weak to strong
static std::atomic<int> s_count_successful_conversions(0);
static std::atomic<int> s_count_failed_conversions(0);

static void attempt_make_a_strong_ptr(RCP<int> ptr, int conversionAttempts) {
  while(!s_bAllowThreadsToRun) {} // spin lock the threads so we can trigger them all at once
  for (int n = 0; n < conversionAttempts; ++n) {
    RCP<int> strongPtr = ptr.create_strong(); // may not succeed if all the strong ptrs have died out
    if (strongPtr.access_private_node().is_node_null()) {
      ++s_count_failed_conversions;
    }
    else {
      ++s_count_successful_conversions;
    }
  }
}

TEUCHOS_UNIT_TEST( RCP, mtRCPStrongToStrong )
{
  const int numThreads = 2;
  const int numConversionAttemptsPerThread = 10000;
  s_count_successful_conversions = 0;
  s_count_failed_conversions = 0;
  try {
    RCP<int> ptr(new int);												// only 1 new allocation happens in this test
    s_bAllowThreadsToRun = false; 										// prepare to spin lock the threads
    std::vector<std::thread> threads;
    for (int threadIndex = 0; threadIndex < numThreads; ++threadIndex) {
      threads.push_back(std::thread(attempt_make_a_strong_ptr, ptr.create_strong(), numConversionAttemptsPerThread));	// all strong
    }
    ptr = null;															// at this point threads are spin locked and holding copies - release the ptr in the main thread
    s_bAllowThreadsToRun = true;										// now we release all the threads
    for (int i = 0; i < threads.size(); ++i) {
      threads[i].join();												// when join completes rcp should be completely deleted
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  int totalAttempts = numConversionAttemptsPerThread * numThreads;
  TEST_EQUALITY_CONST(s_count_failed_conversions, 0);					// all strong so none should fail
  TEST_EQUALITY_CONST(s_count_successful_conversions, totalAttempts);	// all strong so all should succeed
}

TEUCHOS_UNIT_TEST( RCP, mtRCPWeakToStrong )
{
  const int numThreads = 2;
  const int numConversionAttemptsPerThread = 10000;
  s_count_successful_conversions = 0;
  s_count_failed_conversions = 0;
  try {
    RCP<int> ptr(new int);												// only 1 new allocation happens in this test
    s_bAllowThreadsToRun = false; 										// prepare to spin lock the threads
    std::vector<std::thread> threads;
    for (int threadIndex = 0; threadIndex < numThreads; ++threadIndex) {
      threads.push_back(std::thread(attempt_make_a_strong_ptr, ptr.create_weak(), numConversionAttemptsPerThread));	// all weak
    }
    ptr = null;															// at this point threads are spin locked and holding copies - release the ptr in the main thread
    s_bAllowThreadsToRun = true;										// now we release all the threads
    for (int i = 0; i < threads.size(); ++i) {
      threads[i].join();												// when join completes rcp should be completely deleted
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  int totalAttempts = numConversionAttemptsPerThread * numThreads;
  TEST_EQUALITY_CONST(s_count_failed_conversions, totalAttempts);		// all weak so all should fail
  TEST_EQUALITY_CONST(s_count_successful_conversions, 0);				// all weak so none should succeed
}

TEUCHOS_UNIT_TEST( RCP, mtRCPMixedWeakOrStrongToStrong )
{
  const int numThreads = 2;
  const int numConversionAttemptsPerThread = 10000;
  s_count_successful_conversions = 0;
  s_count_failed_conversions = 0;
  try {
    RCP<int> ptr(new int);												// only 1 new allocation happens in this test
    s_bAllowThreadsToRun = false; 										// prepare to spin lock the threads
    std::vector<std::thread> threads;
    bool bCycleStrong = false;
    for (int threadIndex = 0; threadIndex < numThreads; ++threadIndex) {
      threads.push_back(std::thread(attempt_make_a_strong_ptr, bCycleStrong ? ptr.create_strong() : ptr.create_weak(), numConversionAttemptsPerThread));	// all strong
      bCycleStrong = !bCycleStrong;
    }
    ptr = null;															// at this point threads are spin locked and holding copies - release the ptr in the main thread
    s_bAllowThreadsToRun = true;										// now we release all the threads
    for (int i = 0; i < threads.size(); ++i) {
      threads[i].join();												// when join completes rcp should be completely deleted
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  int expectedTotal = numConversionAttemptsPerThread * numThreads;
  std::cout << "Successful Conversions: " << s_count_successful_conversions << std::endl;
  std::cout << "Failed Conversions: " << s_count_failed_conversions << std::endl;

  TEST_INEQUALITY_CONST(s_count_failed_conversions, 0);			// this has to be a mixed result or the test is not doing anything useful
  TEST_INEQUALITY_CONST(s_count_successful_conversions, 0);		// this has to be a mixed result or the test is not doing anything useful
}


//
// Unit Test 1: mtRefCount
// Test reference counting thread safety
// Restore failure by defining BREAK_ATOMIC_USE_FOR_UNSAFE_THREAD_REF_COUNTING at the top - removes the use of the atomics
//
static void make_large_number_of_copies(RCP<int> ptr, int numCopies) {
  std::vector<RCP<int> > ptrs(numCopies, ptr);
}

TEUCHOS_UNIT_TEST( RCP, mtRefCount )
{
  const int numThreads = 4;
  const int numCopiesPerThread = 10000;
  RCP<int> ptr(new int);
  std::vector<std::thread> threads;
  for (int i = 0; i < numThreads; ++i) {
    threads.push_back(std::thread(make_large_number_of_copies, ptr, numCopiesPerThread));
  }
  for (int i = 0; i < threads.size(); ++i) {
    threads[i].join();
  }
  TEST_EQUALITY_CONST(ptr.total_count(), 1);
}



//
// Unit Test 2b: mtNodeTracingWeakNoDealloc
// Related test to determine other problems with the node tracing
// This test demonstrates the necessity of having mutex locks in RCPNodeTracer::getExistingRCPNodeGivenLookupKey for debug mode
// I also have a commented out test here for printActiveRCPNodes - I added thread protection though whether that is ever going to be used like that is unclear.
static void create_independent_weakNoDealloc_rcp_objects(int numAllocations) {
  for(int n = 0; n < numAllocations; ++n ) { // note that getting the second issue to reproduce (the race condition between delete_obj() and removeRCPNode()) may not always hit in this configuration.
    RCP<int> ptr( new int, Teuchos::RCP_WEAK_NO_DEALLOC ); // this allocates a new rcp ptr independent of all other rcp ptrs, and then dumps it, over and over
  }

  // a sub test which would be annoying to leave in - but this can be used to show necessity of locks on the print function - not sure it's a situation that would come up but it is now thread safe
  // Teuchos::RCPNodeTracer::printActiveRCPNodes( std::cout );
}

TEUCHOS_UNIT_TEST( RCP, mtNodeTracingWeakNoDealloc )
{
  const int numThreads = 16;
  const int numRCPAllocations = 10000;
  try {
    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i) {
      threads.push_back(std::thread(create_independent_weakNoDealloc_rcp_objects, numRCPAllocations));
    }
    for (int i = 0; i < threads.size(); ++i) {
      threads[i].join();
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
}

//
// Unit Test 3: mtRCPLastReleaseByAThread
// Test the RCP deletion when the main thread has released - so one of the threads will trigger the actual final delete
// Restore failure by defining BREAK_THREAD_SAFETY_OF_DEINCR_COUNT at the top which restores deincr_count() to a form which is not atomic.
//
static void thread_gets_a_copy_of_rcp(RCP<CatchMemoryLeak> ptr) {
  while(!s_bAllowThreadsToRun) {} // spin lock the threads so we can trigger them all at once
  // note we don't actually do anything - the thread was passed a copy which is all we need for this test - it will be deleted when the thread ends.
  // actually the implementation of thread causes two copies of ptr to exist which can be checked by looking at ptr.strong_count()
}

TEUCHOS_UNIT_TEST( RCP, mtRCPLastReleaseByAThread )
{
  const int numThreads = 16;
  const int numCycles = 5000;												// suitable on my mac in release mode - triggering this race condition requires many cycles
  try {
    for(int cycleIndex = 0; cycleIndex < numCycles; ++cycleIndex) {
    	CatchMemoryLeak::s_countAllocated = 0;								// initialize
    	RCP<CatchMemoryLeak> ptr(new CatchMemoryLeak);						// only 1 new allocation happens in this test
    	s_bAllowThreadsToRun = false; 										// prepare to spin lock the threads
        std::vector<std::thread> threads;
    	for (int threadIndex = 0; threadIndex < numThreads; ++threadIndex) {
    	  threads.push_back(std::thread(thread_gets_a_copy_of_rcp, ptr));
    	}
    	ptr = null;															// at this point threads are spin locked and holding copies - release the ptr in the main thread
    	s_bAllowThreadsToRun = true;										// now we release all the threads
    	for (int i = 0; i < threads.size(); ++i) {
    	  threads[i].join();												// when join completes rcp should be completely deleted
    	}
    	convenience_log_progress(cycleIndex, numCycles);					// this is just output
    	if (CatchMemoryLeak::s_countAllocated != 0) {
    		break; // will catch in error below
    	}
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  TEST_EQUALITY_CONST(CatchMemoryLeak::s_countAllocated, 0);				// test for valid RCP deletion
}

//
// Unit Test 3b: mtRCPLastReleaseByAThreadUsingSetNull
// A variation - let's call ptr = null in the thread (instead of letting it just die at the end of the thread)
//
static void thread_sets_copy_of_rcp_to_null(RCP<CatchMemoryLeak> ptr) {
  while(!s_bAllowThreadsToRun) {} // spin lock the threads so we can trigger them all at once
  ptr = null;
}

TEUCHOS_UNIT_TEST( RCP, mtRCPLastReleaseByAThreadUsingSetNull )
{
  const int numThreads = 16;
  const int numCycles = 5000;												// suitable on my mac in release mode - triggering this race condition requires many cycles
  try {
    for(int cycleIndex = 0; cycleIndex < numCycles; ++cycleIndex) {
    	CatchMemoryLeak::s_countAllocated = 0;								// initialize
    	RCP<CatchMemoryLeak> ptr(new CatchMemoryLeak);						// only 1 new allocation happens in this test
    	s_bAllowThreadsToRun = false; 										// prepare to spin lock the threads
        std::vector<std::thread> threads;
    	for (int threadIndex = 0; threadIndex < numThreads; ++threadIndex) {
    	  threads.push_back(std::thread(thread_sets_copy_of_rcp_to_null, ptr));
    	}
    	ptr = null;															// at this point threads are spin locked and holding copies - release the ptr in the main thread
    	s_bAllowThreadsToRun = true;										// now we release all the threads
    	for (int i = 0; i < threads.size(); ++i) {
    	  threads[i].join();												// when join completes rcp should be completely deleted
    	}
    	convenience_log_progress(cycleIndex, numCycles);					// this is just output
    	if (CatchMemoryLeak::s_countAllocated != 0) {
    		break; // will catch in error below
    	}
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  TEST_EQUALITY_CONST(CatchMemoryLeak::s_countAllocated, 0);				// test for valid RCP deletion
}

//
// Unit Test 4: mtRCPLastReleaseByAThreadWithDealloc
// This is the same as Test 3 except we have added a dealloc
// On initial trials this seems fine - this probably makes sense because as long as node deletion is handled properly, this should also be ok
//
void deallocCatchMemoryLeak(CatchMemoryLeak* ptr) // create a dealloc to go with our CatchMemoryLeak test class
{
  ++CatchMemoryLeak::s_countDeallocs;
  delete ptr;
}

TEUCHOS_UNIT_TEST( RCP, mtRCPLastReleaseByAThreadWithDealloc )
{
  const int numThreads = 16;
  const int numCycles = 1000; // currently arbitrary! This test did not yet reveal a problem so there is no criteria for setting this
  try {
    for(int cycleIndex = 0; cycleIndex < numCycles; ++cycleIndex) {
        CatchMemoryLeak::s_countDeallocs = 0; // set it to 0
    	RCP<CatchMemoryLeak> ptr = rcpWithDealloc(new CatchMemoryLeak, Teuchos::deallocFunctorDelete<CatchMemoryLeak>(deallocCatchMemoryLeak));
    	s_bAllowThreadsToRun = false; 										// prepare to spin lock the threads
        std::vector<std::thread> threads;
    	for (int threadIndex = 0; threadIndex < numThreads; ++threadIndex) {
    	  threads.push_back(std::thread(thread_gets_a_copy_of_rcp, ptr));
    	}
    	ptr = null;															// at this point threads are spin locked and holding copies - release the ptr in the main thread
    	s_bAllowThreadsToRun = true;										// now we release all the threads
    	for (int i = 0; i < threads.size(); ++i) {
    	  threads[i].join();												// when join completes rcp should be completely deleted
    	}
    	convenience_log_progress(cycleIndex, numCycles);			// this is just output
    	if (CatchMemoryLeak::s_countDeallocs != 1) {
    		break; // will catch in error below
    	}
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  TEST_EQUALITY_CONST(CatchMemoryLeak::s_countDeallocs, 1);					// we should have ended with exactly one dealloc call
}

//
// Unit Test 5: mtRCPThreadCallsRelease
// In this test we call release() once from one of the threads - we should be left with a single allocation at the end (a memory leak) to be handled manually
// In initial tests we did not determine there was any problem with release()
//
static void call_release_on_rcp_if_flag_is_set(RCP<CatchMemoryLeak> ptr, int numCopies, bool bCallsRelease) {
  while(!s_bAllowThreadsToRun) {} // spin lock the threads so we can trigger them all at once
  if(bCallsRelease) {
    ptr.release(); // should make a memory leak!
  }
}

TEUCHOS_UNIT_TEST( RCP, mtRCPThreadCallsRelease )
{
  const int numThreads = 16;
  const int numCycles = 1000; // currently arbitrary! This test did not yet reveal a problem so there is no criteria for setting this
  bool bFailure = false;
  try {
    for(int cycleIndex = 0; cycleIndex < numCycles; ++cycleIndex) {
        CatchMemoryLeak::s_countAllocated = 0;								// initialize
    	CatchMemoryLeak * pMemoryToLeak = new CatchMemoryLeak;
    	RCP<CatchMemoryLeak> ptr(pMemoryToLeak);
    	s_bAllowThreadsToRun = false; 										// prepare to spin lock the threads
        std::vector<std::thread> threads;
    	for (int threadIndex = 0; threadIndex < numThreads; ++threadIndex) {
    	  bool bCallRelease = (threadIndex==0); 							// only the first one calls release
    	  threads.push_back(std::thread(call_release_on_rcp_if_flag_is_set, ptr, 1, bCallRelease));
    	}
    	ptr = null;
    	s_bAllowThreadsToRun = true;
    	for (int i = 0; i < threads.size(); ++i) {
    	  threads[i].join();
    	}
    	convenience_log_progress(cycleIndex, numCycles);					// this is just output
    	if (CatchMemoryLeak::s_countAllocated != 1) {
    		break; // will catch in error below
    	}
    	else {
    	  delete pMemoryToLeak; 											// we can clean up our memory leak now by hand - if the test is passing we want a clean finish
    	}
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  TEST_EQUALITY_CONST(bFailure, false);										// organized like this because I want the above loops to properly clean up memory if the test is passing - which changes the counters and makes this interpretation complicated
}

//
// Unit Test 6: mtRCP_getOptionalEmbeddedObj_null
// In this test we use the extra data feature and release from a thread
// Note this could be broken by calling set_extra_data in the threads - but I think that is not the intention
//

template<typename T>
class ExtraDataTest {
public:
  static RCP<ExtraDataTest<T> > create(T *ptr)
    { return rcp(new ExtraDataTest(ptr)); }
  ~ExtraDataTest() { delete [] ptr_; }
private:
  T *ptr_;
  ExtraDataTest(T *ptr) : ptr_(ptr) {}
  // Not defined!
  ExtraDataTest();
  ExtraDataTest(const ExtraDataTest&);
  ExtraDataTest& operator=(const ExtraDataTest&);
};

TEUCHOS_UNIT_TEST( RCP, mtRCPExtraData )
{
  const int numThreads = 16;
  const int numCycles = 1000; // currently arbitrary! This test did not yet reveal a problem so there is no criteria for setting this
  try {
    for(int cycleIndex = 0; cycleIndex < numCycles; ++cycleIndex) {
    	CatchMemoryLeak::s_countAllocated = 0;								// initialize

    	RCP<CatchMemoryLeak> ptr(new CatchMemoryLeak[1]);					// standard delete will be wrong - should call delete[]
    	ptr.release();		// extra data will handle the case
    	Teuchos::set_extra_data( ExtraDataTest<CatchMemoryLeak>::create(ptr.getRawPtr()), "dealloc", Teuchos::inOutArg(ptr));

    	s_bAllowThreadsToRun = false; 										// prepare to spin lock the threads
        std::vector<std::thread> threads;
    	for (int threadIndex = 0; threadIndex < numThreads; ++threadIndex) {
    	  threads.push_back(std::thread(thread_gets_a_copy_of_rcp, ptr));
    	}
    	ptr = null;															// at this point threads are spin locked and holding copies - release the ptr in the main thread
    	s_bAllowThreadsToRun = true;										// now we release all the threads
    	for (int i = 0; i < threads.size(); ++i) {
    	  threads[i].join();												// when join completes rcp should be completely deleted
    	}
    	convenience_log_progress(cycleIndex, numCycles);					// this is just output
    	if (CatchMemoryLeak::s_countAllocated != 0) {
    		break; // will catch in error below
    	}
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  TEST_EQUALITY_CONST(CatchMemoryLeak::s_countAllocated, 0);				// test for valid RCP deletion
}

//
// Unit Test 7: mtRCPDanglingWeak
// Send both strong and weak ptrs to the threads
// Set the main thread to null and let weak/strong race to finish - in the original form we had
//
TEUCHOS_UNIT_TEST( RCP, mtRCPDanglingWeak )
{
  const int numThreads = 1;
  const int numCycles = 5000;												// suitable on my mac in release mode - triggering this race condition requires many cycles
  try {
    for(int cycleIndex = 0; cycleIndex < numCycles; ++cycleIndex) {
    	CatchMemoryLeak::s_countAllocated = 0;								// initialize
    	RCP<CatchMemoryLeak> ptr(new CatchMemoryLeak);						// only 1 new allocation happens in this test
    	s_bAllowThreadsToRun = false; 										// prepare to spin lock the threads
        std::vector<std::thread> threads;
        bool bToggleStrong = true;
    	for (int threadIndex = 0; threadIndex < numThreads; ++threadIndex) {
    	  threads.push_back(std::thread(thread_gets_a_copy_of_rcp, bToggleStrong ? ptr.create_strong() : ptr.create_weak()));
    	  bToggleStrong = !bToggleStrong;
    	}
    	ptr = null;															// at this point threads are spin locked and holding copies - release the ptr in the main thread
    	s_bAllowThreadsToRun = true;										// now we release all the threads
    	for (int i = 0; i < threads.size(); ++i) {
    	  threads[i].join();												// when join completes rcp should be completely deleted
    	}
    	convenience_log_progress(cycleIndex, numCycles);					// this is just output
    	if (CatchMemoryLeak::s_countAllocated != 0) {
    		break; // will catch in error below
    	}
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  TEST_EQUALITY_CONST(CatchMemoryLeak::s_countAllocated, 0);				// test for valid RCP deletion
}

//
// Unit Test 7b: mtRCPWeakToStrong
// Send both strong and weak ptrs to the threads
// Set the main thread to null and let weak/strong race to finish - in the original form we had
//

// Now try the dealloHandle for good measure - for same reason as dealloc, this seems fine
void deallocHandleCatchMemoryLeak(CatchMemoryLeak** handle)
{
  ++CatchMemoryLeak::s_countDeallocs;
  CatchMemoryLeak *ptr = *handle;
  delete ptr;
  *handle = 0;
}

TEUCHOS_UNIT_TEST( RCP, mtRCPLastReleaseByAThreadWithDeallocHandle )
{
  const int numThreads = 16;
  const int numCycles = 1000; // currently arbitrary! This test did not yet reveal a problem so there is no criteria for setting this
  try {
    for(int cycleIndex = 0; cycleIndex < numCycles; ++cycleIndex) {
        CatchMemoryLeak::s_countDeallocs = 0; // set it to 0
    	RCP<CatchMemoryLeak> ptr = rcpWithDealloc(new CatchMemoryLeak, Teuchos::deallocFunctorHandleDelete<CatchMemoryLeak>(deallocHandleCatchMemoryLeak));
    	s_bAllowThreadsToRun = false;										// prepare the threads to be spin locked
        std::vector<std::thread> threads;
    	for (int threadIndex = 0; threadIndex < numThreads; ++threadIndex) {
    	  threads.push_back(std::thread(thread_gets_a_copy_of_rcp, ptr));
    	}
    	ptr = null;															// at this point threads are spin locked and holding copies - release the ptr in the main thread
    	s_bAllowThreadsToRun = true;										// now we release all the threads
    	for (int i = 0; i < threads.size(); ++i) {
    	  threads[i].join();												// when join completes rcp should be completely deleted
    	}
    	convenience_log_progress(cycleIndex, numCycles);					// this is just output
    	if (CatchMemoryLeak::s_countDeallocs != 1) {
    		break; // will catch in error below
    	}
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  TEST_EQUALITY_CONST(CatchMemoryLeak::s_countDeallocs, 1);					// should be 1 deallocHandle call
}
#endif // end #ifndef KNOCK_OUT_ALL_CODE

} // namespace

#endif // HAVE_TEUCHOSCORE_CXX11


