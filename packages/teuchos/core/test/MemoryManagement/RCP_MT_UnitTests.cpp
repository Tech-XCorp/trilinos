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

// #define DISABLE_ATOMIC_COUNTERS       				// breaks test 1 by changing atomics to ints
// To break Test 2 comment out the following in Teuchos RCPNode.cpp: #define DISABLE_MUTEX_WHICH_PROTECTS_DEBUG_NODE_TRACING
// To break Test 2b comment out the mutex locks and unlocks in RCPNodeTracer::getExistingRCPNodeGivenLookupKey()
// #define BREAK_THREAD_SAFETY_OF_DEINCR_COUNT			// breaks test 3 by changing strong decrement to be non-atomic
// #define INTRODUCE_RACE_CONDITIONS_FOR_UNBINDING	// breaks test 7 by introducing a race condition so weak mixed with strong can crash
// #define BREAK_ATOMIC_WEAK_TO_STRONG_CONVERSION     // breaks test 9 by making weak to strong conversion not thread safe

#include "TeuchosCore_ConfigDefs.hpp"
#include "General_MT_UnitTests.hpp"

#ifdef HAVE_TEUCHOSCORE_CXX11
#include "Teuchos_RCP.hpp"
#include "Teuchos_RCPNode.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include <vector>
#include <thread>

namespace {
using Teuchos::null;
using Teuchos::RCP;
using Teuchos::rcp;

//
// Unit Test 1: mtRefCount
// Test reference counting thread safety
// Restore failure by #define DISABLE_ATOMIC_COUNTERS
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
// Unit Test 2: mtCreateIndependentRCP
// Test debug node tracing thread safety
// Restore failure by defining DISABLE_MUTEX_WHICH_PROTECTS_DEBUG_NODE_TRACING at the top (and run debug mode) which removes protective mutex on RCPNodeTracer::removeRCPNode() and RCPNodeTracer::addNewRCPNode()
//
static std::atomic<bool> s_bAllowThreadsToRun;				// this static is used to spin lock all the threads - after we allocate we set this true and all threads can complete

static void create_independent_rcp_objects(int numAllocations) {
  while (!s_bAllowThreadsToRun) {}
  for(int n = 0; n < numAllocations; ++n ) {
    RCP<int> ptr( new int ); // this allocates a new rcp ptr independent of all other rcp ptrs, and then dumps it, over and over
  }
}

TEUCHOS_UNIT_TEST( RCP, mtCreateIndependentRCP )
{
  const int numThreads = 4;
  const int numRCPAllocations = 10000;
#ifndef TEUCHOS_DEBUG
  std::cout << "Release Mode - This test was designed to solve a Debug Mode problem." << std::endl;
#endif
  int initialNodeCount = Teuchos::RCPNodeTracer::numActiveRCPNodes();
  try {
    std::vector<std::thread> threads;
    s_bAllowThreadsToRun = false;
    for (int i = 0; i < numThreads; ++i) {
      threads.push_back(std::thread(create_independent_rcp_objects, numRCPAllocations));
    }
    s_bAllowThreadsToRun = true;
    for (int i = 0; i < threads.size(); ++i) {
      threads[i].join();
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  TEST_EQUALITY_CONST(initialNodeCount, Teuchos::RCPNodeTracer::numActiveRCPNodes());
}

//
// Unit Test 2b: mtTestGetExistingRCPNodeGivenLookupKey
// Related test to determine other problems with the node tracing
// This test demonstrates the necessity of having mutex locks in RCPNodeTracer::getExistingRCPNodeGivenLookupKey for debug mode
// Using Teuchos::RCP_WEAK_NO_DEALLOC to construct will also cause problems with node tracing if not protected by the mutex
static void create_independent_rcp_without_ownership(int numAllocations) {
  for(int n = 0; n < numAllocations; ++n ) { // note that getting the second issue to reproduce (the race condition between delete_obj() and removeRCPNode()) may not always hit in this configuration.
    int * intPtr = new int;
    RCP<int> ptr( intPtr, false ); // this allocates a new rcp independent of all other rcp ptrs, and then dumps it, over and over
    delete intPtr;
  }
}

TEUCHOS_UNIT_TEST( RCP, mtTestGetExistingRCPNodeGivenLookupKey )
{
  const int numThreads = 4;
  const int numRCPAllocations = 50000;
  try {
    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i) {
      threads.push_back(std::thread(create_independent_rcp_without_ownership, numRCPAllocations));
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
template<typename SOURCE_RCP_TYPE>
static void thread_gets_a_copy_of_rcp(SOURCE_RCP_TYPE ptr) {
  while(!s_bAllowThreadsToRun) {} // spin lock the threads so we can trigger them all at once
  // note we don't actually do anything - the thread was passed a copy which is all we need for this test - it will be deleted when the thread ends.
}

TEUCHOS_UNIT_TEST( RCP, mtRCPLastReleaseByAThread )
{
  const int numThreads = 4;
  const int numCycles = 5000;												// suitable on my mac in release mode - triggering this race condition requires many cycles
  try {
    for(int cycleIndex = 0; cycleIndex < numCycles; ++cycleIndex) {
      CatchMemoryLeak::s_countAllocated = 0;								// initialize
      RCP<CatchMemoryLeak> ptr(new CatchMemoryLeak);						// only 1 new allocation happens in this test
      s_bAllowThreadsToRun = false; 										// prepare to spin lock the threads
      std::vector<std::thread> threads;
      for (int threadIndex = 0; threadIndex < numThreads; ++threadIndex) {
        threads.push_back(std::thread(thread_gets_a_copy_of_rcp<RCP<CatchMemoryLeak>>, ptr));
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
// Perhaps we can delete this one.
//
static void thread_sets_copy_of_rcp_to_null(RCP<CatchMemoryLeak> ptr) {
  while(!s_bAllowThreadsToRun) {} // spin lock the threads so we can trigger them all at once
  ptr = null;
}

TEUCHOS_UNIT_TEST( RCP, mtRCPLastReleaseByAThreadUsingSetNull )
{
  const int numThreads = 4;
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
  const int numThreads = 4;
  const int numCycles = 5000; // currently arbitrary! This test did not yet reveal a problem so there is no criteria for setting this
  try {
    for(int cycleIndex = 0; cycleIndex < numCycles; ++cycleIndex) {
      CatchMemoryLeak::s_countDeallocs = 0; // set it to 0
      RCP<CatchMemoryLeak> ptr = rcpWithDealloc(new CatchMemoryLeak, Teuchos::deallocFunctorDelete<CatchMemoryLeak>(deallocCatchMemoryLeak));
      s_bAllowThreadsToRun = false; 										// prepare to spin lock the threads
      std::vector<std::thread> threads;
      for (int threadIndex = 0; threadIndex < numThreads; ++threadIndex) {
        threads.push_back(std::thread(thread_gets_a_copy_of_rcp<RCP<CatchMemoryLeak>>, ptr));
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
  const int numThreads = 4;
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
// Note this could be broken by calling set_extra_data in the threads
// Probably it's always set at constructor but we can mutex protect this

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
  const int numThreads = 4;
  const int numCycles = 5000; // currently arbitrary! This test did not yet reveal a problem so there is no criteria for setting this
  try {
    for(int cycleIndex = 0; cycleIndex < numCycles; ++cycleIndex) {
      CatchMemoryLeak::s_countAllocated = 0;								// initialize
      RCP<CatchMemoryLeak> ptr(new CatchMemoryLeak[1]);					// standard delete will be wrong - should call delete[]
      ptr.release();		// extra data will handle the case
      Teuchos::set_extra_data( ExtraDataTest<CatchMemoryLeak>::create(ptr.getRawPtr()), "dealloc", Teuchos::inOutArg(ptr));
      s_bAllowThreadsToRun = false; 										// prepare to spin lock the threads
      std::vector<std::thread> threads;
      for (int threadIndex = 0; threadIndex < numThreads; ++threadIndex) {
        threads.push_back(std::thread(thread_gets_a_copy_of_rcp<RCP<CatchMemoryLeak>>, ptr));
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
// Unit Test 7: mtRCPWeakStrongDeleteRace
// Send both strong and weak ptrs to the threads - Set the main thread to null and let weak/strong race to finish and delete
// #define INTRODUCE_RACE_CONDITIONS_FOR_UNBINDING to break this and create conditions similar to original where a weak could race and delete node while object was still finishing up
//
TEUCHOS_UNIT_TEST( RCP, mtRCPWeakStrongDeleteRace )
{
  const int numThreads = 4;
  const int numCycles = 5000;												// suitable on my mac in release mode - triggering this race condition requires many cycles
  try {
    for(int cycleIndex = 0; cycleIndex < numCycles; ++cycleIndex) {
      CatchMemoryLeak::s_countAllocated = 0;								// initialize
      RCP<CatchMemoryLeak> ptr(new CatchMemoryLeak);						// only 1 new allocation happens in this test
      s_bAllowThreadsToRun = false; 										// prepare to spin lock the threads
      std::vector<std::thread> threads;
      bool bToggleStrong = true;
      for (int threadIndex = 0; threadIndex < numThreads; ++threadIndex) {
        if (bToggleStrong) {
          threads.push_back(std::thread(thread_gets_a_copy_of_rcp<RCP<CatchMemoryLeak>>, ptr.create_strong()));
        }
        else {
          threads.push_back(std::thread(thread_gets_a_copy_of_rcp<RCP<CatchMemoryLeak>>, ptr.create_weak()));
        }
        bToggleStrong = !bToggleStrong;
      }
      ptr = null;
      // at this point threads are spin locked and holding copies - release the ptr in the main thread
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
// Unit Test 8: mtRCPLastReleaseByAThreadWithDeallocHandle
// Now try the dealloHandle for good measure - for same reason as dealloc, this seems fine
// We can remove this test I think it's probably redundant to the dealloc test.
//
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
        threads.push_back(std::thread(thread_gets_a_copy_of_rcp<RCP<CatchMemoryLeak>>, ptr));
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

//
// Unit Test 9: mtRCPMixedWeakAndStrongConvertToStrong
// This test will force conversions to strong from weak while the strong pool is deleting
// It will generate a series of successful conversions which will then begin to return null when strong is gone
// break this with #define BREAK_ATOMIC_WEAK_TO_STRONG_CONVERSION
//
static std::atomic<int> s_count_successful_conversions(0);
static std::atomic<int> s_count_failed_conversions(0);

template<class SOURCE_RCP_TYPE>
static void attempt_make_a_strong_ptr(SOURCE_RCP_TYPE ptr) {
  while(!s_bAllowThreadsToRun) {} // spin lock the threads so we can trigger them all at once
  RCP<CatchMemoryLeak> possibleStrongPtr = ptr.create_strong(true); // ptr can be weak or strong - the weak ptrs may fail
  if (possibleStrongPtr.is_null()) {
    ++s_count_failed_conversions;
  }
  else {
    ++s_count_successful_conversions;
  }
}

TEUCHOS_UNIT_TEST( RCP, mtRCPMixedWeakAndStrongConvertToStrong )
{
  const int numThreads = 4;
  int numCycles = 10000;
  s_count_successful_conversions = 0;
  s_count_failed_conversions = 0;
  try {
    for(int cycleIndex = 0; cycleIndex < numCycles; ++cycleIndex) {
      CatchMemoryLeak::s_countAllocated = 0;								// initialize
      RCP<CatchMemoryLeak> ptr(new CatchMemoryLeak);						// only 1 new allocation happens in this test
      s_bAllowThreadsToRun = false; 										// prepare to spin lock the threads
      std::vector<std::thread> threads;
      bool bCycleStrong = true;
      for (int threadIndex = 0; threadIndex < numThreads; ++threadIndex) {
        if (bCycleStrong) {
          threads.push_back(std::thread(attempt_make_a_strong_ptr<RCP<CatchMemoryLeak>>, ptr.create_strong()));
        }
        else {
          threads.push_back(std::thread(attempt_make_a_strong_ptr<RCP<CatchMemoryLeak>>, ptr.create_weak()));
        }
        bCycleStrong = !bCycleStrong;
      }
      ptr = null;															// at this point threads are spin locked and holding copies - release the ptr in the main thread
      s_bAllowThreadsToRun = true;											// now we release all the threads
      for (int i = 0; i < threads.size(); ++i) {
        threads[i].join();													// when join completes rcp should be completely deleted
      }
      if (CatchMemoryLeak::s_countAllocated != 0) {
        break;
      }
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  std::cout << "WeakRCP->RCP conversion returned null " << s_count_failed_conversions << " times and succeeded " << s_count_successful_conversions << " times. We want to see some of each in this test." << std::endl;

  TEST_INEQUALITY_CONST(s_count_failed_conversions, 0);			// this has to be a mixed result or the test is not doing anything useful
  TEST_INEQUALITY_CONST(s_count_successful_conversions, 0);		// this has to be a mixed result or the test is not doing anything useful
  TEST_EQUALITY(CatchMemoryLeak::s_countAllocated, 0); 			// should be 0
  

}

} // namespace

#endif // HAVE_TEUCHOSCORE_CXX11
