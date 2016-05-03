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

//#define BREAK_ATOMIC_USE_FOR_UNSAFE_THREAD_REF_COUNTING		// causes unit test 1 (mtRefCount) to have failures - restores the original non-atomic counters
// In Teuchos_RCPNode.cpp comment back in BREAK_MUTEX_WHICH_PROTECTS_DEBUG_NODE_TRACING	- causes unit test 2 (mtCreatedIndependentRCP) to have node tracing failures (debug mode only)
//#define BREAK_THREAD_SAFETY_OF_DEINCR_COUNT					// causes unit test 3 to fail by breaking deincr_count() in Teuchos_RCPNode.hpp

// #define TRACK_TOTAL_COUNT_NOT_WEAK_COUNT

#include "TeuchosCore_ConfigDefs.hpp"

#ifdef HAVE_TEUCHOSCORE_CXX11

#include "Teuchos_RCP.hpp"
#include "Teuchos_getConst.hpp"
#include "Teuchos_getBaseObjVoidPtr.hpp"
#include "Teuchos_RCPStdSharedPtrConversions.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

#include <vector>
#include <thread>
#include <stack>

#include "TestClasses.hpp"
#include "Teuchos_UnitTestHarness.hpp"

namespace {

using Teuchos::as;
using Teuchos::null;
using Teuchos::Ptr;
using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::rcpFromRef;
using Teuchos::rcpFromUndefRef;
using Teuchos::outArg;
using Teuchos::rcpWithEmbeddedObj;

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
// Unit Test 2: mtCreatedIndependentRCP
// Test debug node tracing thread safety
// Restore failure by defining BREAK_MUTEX_WHICH_PROTECTS_DEBUG_NODE_TRACING at the top (and run debug mode) which removes protective mutex on RCPNodeTracer::removeRCPNode() and RCPNodeTracer::addNewRCPNode()
//
static void create_independent_rcp_objects(int numAllocations) {
  for(int n = 0; n < numAllocations; ++n ) { // note that getting the second issue to reproduce (the race condition between delete_obj() and removeRCPNode()) may not always hit in this configuration.
    RCP<int> ptr( new int ); // this allocates a new rcp ptr independent of all other rcp ptrs, and then dumps it, over and over
  }
}

TEUCHOS_UNIT_TEST( RCP, mtCreatedIndependentRCP )
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

//
// Unit Test 3: mtRCPLastReleaseByAThread
// Test the RCP deletion when the main thread has released - so one of the threads will trigger the actual final delete
// Restore failure by defining BREAK_THREAD_SAFETY_OF_DEINCR_COUNT at the top which restores deincr_count() to a form which is not atomic.
//
class CatchMemoryLeak // This class is a utility class which tracks constructor/destructor calls (for this test) or counts times a dealloc or deallocHandle was implemented (for later tests)
{
public:
	CatchMemoryLeak() { ++s_countAllocated; }
	~CatchMemoryLeak() { --s_countAllocated; }
	static std::atomic<int> s_countAllocated;
	static std::atomic<int> s_countDeallocs;
};
std::atomic<int> CatchMemoryLeak::s_countAllocated(0);	// counts constructor calls (+1) and destructor calls (-1) which may include double delete events
std::atomic<int> CatchMemoryLeak::s_countDeallocs(0);	// counts dealloc or dellocHandle calls - used for test 4 and test 5

static std::atomic<bool> s_bAllowThreadsToRun;				// this static is used to spin lock all the threads - after we allocate we set this true and all threads can complete

static void thread_gets_a_copy_of_rcp(RCP<CatchMemoryLeak> ptr) {
  while(!s_bAllowThreadsToRun) {} // spin lock the threads so we can trigger them all at once
  // note we don't actually do anything - the thread was passed a copy which is all we need for this test - it will be deleted when the thread ends.
  // actually the implementation of thread causes two copies of ptr to exist which can be checked by looking at ptr.strong_count()
}

// this is a convenience function for outputting percent complete information for long tests designed to find race conditions
static void convenience_log_progress(int cycle, int totalCycles, bool bEnding) {
  if (cycle==0) {
    std::cout << "Percent complete: ";												// begin the log line
  }
  int mod = (totalCycles/10);														// log every 10% percent complete - using mod % to output at regular intervals
  if((cycle % (mod == 0 ? 1 : mod) == 0) || (cycle == totalCycles-1)) {				// sometimes quick testing so make sure mod is not 0
    std::cout << (int)( 100.0f * (float) cycle / (float) (totalCycles-1) ) << "% ";
  }
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
    	bool bFailure = (CatchMemoryLeak::s_countAllocated != 0);			// verify the counter is back to 0 and break for failure if it is not
    	convenience_log_progress(cycleIndex, numCycles, bFailure);			// this is just output
    	if (bFailure) {
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
    	bool bFailure = (CatchMemoryLeak::s_countDeallocs != 1);			// verify the counter is back to 0 and break for failure if it is not
    	convenience_log_progress(cycleIndex, numCycles, bFailure);			// this is just output
    	if (bFailure) {
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
    	bFailure = (CatchMemoryLeak::s_countAllocated != 1);			    // verify the counter is 1 (due to memory leak) and break for failure if it is not
    	convenience_log_progress(cycleIndex, numCycles, bFailure);			// this is just output
    	if (bFailure) {
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

    	// Option 2 - incorrect allocation, no extra data - this should trigger a fail event
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
    	bool bFailure = (CatchMemoryLeak::s_countAllocated != 0);			// verify the counter is back to 0 and break for failure if it is not
    	convenience_log_progress(cycleIndex, numCycles, bFailure);			// this is just output
    	if (bFailure) {
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
    	bool bFailure = (CatchMemoryLeak::s_countAllocated != 0);			// verify the counter is back to 0 and break for failure if it is not
    	convenience_log_progress(cycleIndex, numCycles, bFailure);			// this is just output
    	if (bFailure) {
    		break; // will catch in error below
    	}
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  TEST_EQUALITY_CONST(CatchMemoryLeak::s_countAllocated, 0);				// test for valid RCP deletion
}

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
    	bool bFailure = (CatchMemoryLeak::s_countDeallocs != 1);			// verify the counter is back to 0 and break for failure if it is not
    	convenience_log_progress(cycleIndex, numCycles, bFailure);			// this is just output
    	if (bFailure) {
    		break; // will catch in error below
    	}
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  TEST_EQUALITY_CONST(CatchMemoryLeak::s_countDeallocs, 1);					// should be 1 deallocHandle call
}

} // namespace

#endif // HAVE_TEUCHOSCORE_CXX11
