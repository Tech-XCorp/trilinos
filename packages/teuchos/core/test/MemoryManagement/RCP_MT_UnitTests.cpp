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

#include "TeuchosCore_ConfigDefs.hpp"

#ifdef HAVE_TEUCHOSCORE_CXX11

#include "Teuchos_RCP.hpp"
#include "Teuchos_getConst.hpp"
#include "Teuchos_getBaseObjVoidPtr.hpp"
#include "Teuchos_RCPStdSharedPtrConversions.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

// added this for some sleep functionality to help test threads in some cases - I expect to pull this out eventually
#ifndef ICL
#include <unistd.h>
#else
void sleep(int sec)
{
  Sleep(sec);
}
#endif

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
// Test multi-threaded reference counting
//

static void make_large_number_of_copies(RCP<int> ptr, int numCopies) {
  std::vector<RCP<int> > ptrs(numCopies, ptr);
}

TEUCHOS_UNIT_TEST( RCP, mtRefCount )
{
  RCP<int> ptr(new int);
  TEST_ASSERT(ptr.total_count() == 1);
  std::vector<std::thread> threads;
  for (int i = 0; i < 4; ++i) {
    threads.push_back(std::thread(make_large_number_of_copies, ptr, 10000));
  }
  for (int i = 0; i < threads.size(); ++i) {
    threads[i].join();
  }
  TEST_EQUALITY_CONST(ptr.total_count(), 1);
}

//
// Test Debug node tracing
// This test would fail with various memory problems due to the RCP node tracing not being thread safe
// This test would originally only fail with multiple threads and debug mode
// First fix is to use a single mutex to lock both RCPNodeTracer::removeRCPNode() and RCPNodeTracer::addNewRCPNode()
// Second issue can be harder to reproduce - originally we had node_->delete_obj(), followed by RCPNodeTracer::removeRCPNode()
// However if another thread calls RCPNodeTracer::addNewRCPNode() after the delete_obj() call and before the removeRCPNode() call it can claim the address space

static void create_independent_rcp_objects() {
  for(int n = 0; n < 10000; ++n ) { // note that getting the second issue to reproduce (the race condition between delete_obj() and removeRCPNode()) may not always hit in this configuration.
    RCP<int> ptr( new int ); // this allocates a new rcp ptr independent of all other rcp ptrs, and then dumps it, over and over
  }
}

TEUCHOS_UNIT_TEST( RCP, mtCreatedIndependentRCP )
{
  // first let's establish if we are in Debug mode - this particular test relies on it
  std::cout << std::endl;
  #ifdef TEUCHOS_DEBUG
	std::cout << "Running in Debug Mode. Multi-Threading originally caused the node tracing to fail in Debug." << std::endl;
  #else
	// I think we want a quiet test
	// std::cout << "Running in Release Mode. Multi-Threading did not cause any problems in Release." << std::endl;
  #endif

  std::vector<std::thread> threads;
  int numThreads = 4;
  for (int i = 0; i < numThreads; ++i) {
    threads.push_back(std::thread(create_independent_rcp_objects));
  }
  for (int i = 0; i < threads.size(); ++i) {
    threads[i].join();
  }

  // this will crash if the bug happens - need to determine if there is more I can do to collect.
}

// this test is similar to the first, except now we will set ptr to null in the main thread before threads finish
// this means all threads will finish up, release rcp, and at some point it should be gone
// we want to create a race condition on this delete
// the CatchMemoryLeak used to a simple atomic counter to track allocations and deletes
// I may want to change this to non atomic and just check for deletion of the single object to make sure the existance of that atomic is not changing an error that would otherwise happen
class CatchMemoryLeak
{
public:
	CatchMemoryLeak() { ++s_countAllocated; }
	~CatchMemoryLeak() { --s_countAllocated; }
	static std::atomic<int> s_countAllocated;
};
std::atomic<int> CatchMemoryLeak::s_countAllocated(0);

// To debug this test I store the final RCP count for each thread when the thread completes - this also can go away when this test is finalized
const int kUseThreadCount = 16;
int saveThreadStrongCountAtEnd[kUseThreadCount];

static void make_large_number_of_copies_of_CatchMemoryLeak(RCP<CatchMemoryLeak> ptr, int numCopies, int threadIndex) {
  saveThreadStrongCountAtEnd[threadIndex] = 0;
  if(true) { // we want the scope of the vector to end before we save the count - one of the threads (usually last one) should finish with a count of 2 (if null was written) - the extra one is held by the thread management, something I don't know how it works
	  std::vector<RCP<CatchMemoryLeak> > ptrs(numCopies, ptr);
  }
  saveThreadStrongCountAtEnd[threadIndex] = ptr.strong_count();
}

// The test is currently not very efficiency and may require more than 20000 test cycles to actually trigger
// Clearly this will not be the final form
// The status right now is that we have race conditions in the vicinity of unbindOne()
// The test does not break counters - but memory deallocation fails is the most common case - meaning the race conditions cause the actual delete to be skipped
// This generally appears to be caused by two threads simultaneously arrive to delete (count is 2), both decrement (count is 0), both reincrement (count is 2), then both skip the delete (which looks for count 1)
TEUCHOS_UNIT_TEST( RCP, mtRCPLastReleaseByAThread )
{
  try {
    /* // this code can be modified to confirm that we catch a double delete or a memory leak - the static CatchMemoryLeak::s_countAllocated must be 0 when this test finishes
	CatchMemoryLeak * testPtr = new CatchMemoryLeak();	// creates some memory
    RCP<CatchMemoryLeak> testRCP(testPtr);				// puts it under control of RCP
    testRCP.release();  								// releases control - Remove this to double delete
    delete testPtr;		  								// manually deletes - Remove this to leak
	*/

	int counterOutput = 0;
    for(int cycleIndex = 0; cycleIndex < 100000; ++cycleIndex) {
    	RCP<CatchMemoryLeak> ptr(new CatchMemoryLeak);

        std::vector<std::thread> threads;
    	for (int threadIndex = 0; threadIndex < kUseThreadCount; ++threadIndex) {
    	  threads.push_back(std::thread(make_large_number_of_copies_of_CatchMemoryLeak, ptr, 5000, threadIndex));
    	}
    	// sleep(1); // this will cause the threads to be done and the memory to be released when you start ptr = null - which means you could then see the count actually drop by 1
    	int countBeforeNull = CatchMemoryLeak::s_countAllocated;
    	ptr = null;
    	int countAfterNull = CatchMemoryLeak::s_countAllocated;

    	++counterOutput;
    	if( counterOutput == 100 ) {
    	  std::cout << "Cycle: " << cycleIndex << "  Before null: " << countBeforeNull << "   After null: " << countAfterNull << std::endl;
    	  counterOutput = 0;
    	}

    	for (int i = 0; i < threads.size(); ++i) {
    	  threads[i].join();
    	}

    	if( counterOutput == 0 ) {
  	      for(int threadIndex = 0; threadIndex < kUseThreadCount; ++threadIndex) {
  		    std::cout << saveThreadStrongCountAtEnd[threadIndex] << "   ";
  	      }
  		  std::cout << std::endl;
  	    }

    	if(CatchMemoryLeak::s_countAllocated != 0) {
    		std::cout << "Stopping on cycle: " << cycleIndex << " Remaining Allocations: " << CatchMemoryLeak::s_countAllocated << std::endl;
    		break; // will catch in error below
    	}
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);

  TEST_EQUALITY_CONST(CatchMemoryLeak::s_countAllocated, 0);		// these should match
}

} // namespace

#endif // HAVE_TEUCHOSCORE_CXX11
