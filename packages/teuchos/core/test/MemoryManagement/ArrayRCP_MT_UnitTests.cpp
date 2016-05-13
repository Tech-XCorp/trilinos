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

//#define RUN_ARRAY_RCP_UNIT_TESTS - temporary - disable these tests

#include "TeuchosCore_ConfigDefs.hpp"
#include "General_MT_UnitTests.hpp"

#ifdef RUN_ARRAY_RCP_UNIT_TESTS
#ifdef HAVE_TEUCHOSCORE_CXX11

#include "Teuchos_ArrayRCP.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include <vector>
#include <thread>

namespace {

using Teuchos::null;
using Teuchos::ArrayRCP;
using Teuchos::arcp;

//
// Unit Test 1: mtRefCount
// Test reference counting thread safety
// This is also based on RCPNodeHandle so it's not surprising that it's ok
//
static void make_large_number_of_arrayrcp_copies(ArrayRCP<int> ptr, int numCopies) {
  std::vector<ArrayRCP<int> > ptrs(numCopies, ptr);
}

TEUCHOS_UNIT_TEST( ArrayRCP, mtRefCount )
{
  const int numThreads = 4;
  const int numCopiesPerThread = 10000;
  const int arraySize = 10;

  ArrayRCP<int> ptr = arcp<int>(arraySize);
  std::vector<std::thread> threads;
  for (int i = 0; i < numThreads; ++i) {
    threads.push_back(std::thread(make_large_number_of_arrayrcp_copies, ptr, numCopiesPerThread));
  }
  for (int i = 0; i < threads.size(); ++i) {
    threads[i].join();
  }
  TEST_EQUALITY_CONST(ptr.total_count(), 1);
}

//
// Unit Test 2: mtCreatedIndependentArrayRCP
// Same as 1, this is ok because RCP is now ok and it really tests the same mechanism
// Consider deleting this Test?
//
static void create_independent_arrayrcp_objects(int numAllocations) {
  for(int n = 0; n < numAllocations; ++n ) { // note that getting the second issue to reproduce (the race condition between delete_obj() and removeRCPNode()) may not always hit in this configuration.
    ArrayRCP<int> ptr = arcp<int>(10); // this allocates a new rcp ptr independent of all other rcp ptrs, and then dumps it, over and over
  }
}

TEUCHOS_UNIT_TEST( ArrayRCP, mtCreatedIndependentArrayRCP )
{
  const int numThreads = 16;
  const int numArrayRCPAllocations = 10000;
  try {
    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i) {
      threads.push_back(std::thread(create_independent_arrayrcp_objects, numArrayRCPAllocations));
    }
    for (int i = 0; i < threads.size(); ++i) {
      threads[i].join();
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
}

static std::atomic<bool> s_bAllowThreadsToRun;				// this static is used to spin lock all the threads - after we allocate we set this true and all threads can complete

//
// Unit Test 3: mtResize
// the resize operation will implement a copy if the size changes
// during the thread operation we also set this to null
// this test verifies that resize operations won't mess up threads - but it's not expected they would because it can only copy the root but not change it
static void iterate_on_arrayrcp(ArrayRCP<CatchMemoryLeak> ptr) {
    while(!s_bAllowThreadsToRun) {} // spin lock the threads so we can trigger them all at once
	ptr.resize(ptr.size());
	ptr.resize(ptr.size()+1);
	ptr.resize(ptr.size());
	ptr = null;
}

TEUCHOS_UNIT_TEST( ArrayRCP, mtResize )
{
  const int arrayRCPSizeValue = 100;
  const int numThreads = 16;
  const int numCycles = 1000;
  try {
    for (int cycleIndex = 0; cycleIndex < numCycles; ++cycleIndex) {
      CatchMemoryLeak::s_countAllocated = 0;
      ArrayRCP<CatchMemoryLeak> ptr = arcp<CatchMemoryLeak>(100);
      std::vector<std::thread> threads;
      s_bAllowThreadsToRun = false;
      for (int i = 0; i < numThreads; ++i) {
        threads.push_back(std::thread(iterate_on_arrayrcp, ptr));
      }
      ptr = null; // add additional stress to the test
      s_bAllowThreadsToRun = true; // release the threads
      for (int i = 0; i < threads.size(); ++i) {
        threads[i].join();
      }
      convenience_log_progress(cycleIndex, numCycles);
  	  if (CatchMemoryLeak::s_countAllocated != 0) {
        break; // will catch in error below
  	  }
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  TEST_EQUALITY_CONST(CatchMemoryLeak::s_countAllocated, 0);				// test for valid RCP deletion
}

//
// Unit Test 4: mtRelease
// Study the behavior when we call release on the array
// we should end up with a memory leak equivalent to the size of the array
// at the end of each loop we manually delete the memory leak so the test completes with count back to 0
//
static void call_release_on_arrayRCP(ArrayRCP<CatchMemoryLeak> ptr) {
    while(!s_bAllowThreadsToRun) {} // spin lock the threads so we can trigger them all at once
	ptr.release();
}

TEUCHOS_UNIT_TEST( ArrayRCP, mtRelease )
{
  const int arrayRCPSizeValue = 100;
  const int numThreads = 16;
  const int numCycles = 1000;
  try {
    for (int cycleIndex = 0; cycleIndex < numCycles; ++cycleIndex) {
      CatchMemoryLeak::s_countAllocated = 0;
      CatchMemoryLeak * pArrayWithMemoryLeak = new CatchMemoryLeak[arrayRCPSizeValue];
      ArrayRCP<CatchMemoryLeak> ptr = arcp<CatchMemoryLeak>(pArrayWithMemoryLeak, 0, arrayRCPSizeValue);
      std::vector<std::thread> threads;
      s_bAllowThreadsToRun = false;
      for (int i = 0; i < numThreads; ++i) {
        threads.push_back(std::thread(call_release_on_arrayRCP, ptr));
      }
      ptr = null; // add additional stress to the test
      s_bAllowThreadsToRun = true;
      for (int i = 0; i < threads.size(); ++i) {
        threads[i].join();
      }
      convenience_log_progress(cycleIndex, numCycles);
  	  if (CatchMemoryLeak::s_countAllocated != arrayRCPSizeValue) { // we should have arrayRCPSizeValue count of the class allocated
        break; // will catch in error below
  	  }
  	  delete [] pArrayWithMemoryLeak; // resolve the memory leak
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  TEST_EQUALITY_CONST(CatchMemoryLeak::s_countAllocated, 0);				// test for valid RCP deletion - we should have resolved the memory leak so now this should be 0 again
}

//
// Unit Test 5: mtUseIterator
// Call iterator++ using shared ArrayRCP
// The iterator mechanism is just in ArrayRCP and has nothing to do with the underlying node
// We expect no trouble and this seems seems fine
//
static void use_iterator_on_arrayRCP(ArrayRCP<int> ptr) {
	typename ArrayRCP<int>::const_iterator itr = ptr.begin();
	for( int i = 0; itr < ptr.end(); ++i, ++itr )
	  TEUCHOS_TEST_FOR_EXCEPT( !(*itr == ptr[i]) );
}

TEUCHOS_UNIT_TEST( ArrayRCP, mtUseIterator )
{
  const int numThreads = 4;
  const int arraySize = 10;
  ArrayRCP<int> ptr = arcp<int>(arraySize);
  std::vector<std::thread> threads;
  for (int i = 0; i < numThreads; ++i) {
    threads.push_back(std::thread(use_iterator_on_arrayRCP, ptr));
  }
  for (int i = 0; i < threads.size(); ++i) {
    threads[i].join();
  }
}

//
// Unit Test 6: mtAssign
// Test the assing function from multiple threads
//
static void assign_on_arrayrcp(ArrayRCP<CatchMemoryLeak> ptr) {
    while(!s_bAllowThreadsToRun) {} // spin lock the threads so we can trigger them all at once
	ptr.assign( ptr.size(), CatchMemoryLeak());
	ptr.assign( ptr.size() + 10, CatchMemoryLeak());
	ptr.assign( ptr.size() - 5, CatchMemoryLeak());
	ptr = null;
}

TEUCHOS_UNIT_TEST( ArrayRCP, mtAssign )
{
  const int arrayRCPSizeValue = 100;
  const int numThreads = 16;
  const int numCycles = 1000;
  try {
    for (int cycleIndex = 0; cycleIndex < numCycles; ++cycleIndex) {
      CatchMemoryLeak::s_countAllocated = 0;
      ArrayRCP<CatchMemoryLeak> ptr = arcp<CatchMemoryLeak>(100);
      std::vector<std::thread> threads;
      s_bAllowThreadsToRun = false;
      for (int i = 0; i < numThreads; ++i) {
        threads.push_back(std::thread(assign_on_arrayrcp, ptr));
      }
      ptr = null; // add additional stress to the test
      s_bAllowThreadsToRun = true; // release the threads
      for (int i = 0; i < threads.size(); ++i) {
        threads[i].join();
      }
      convenience_log_progress(cycleIndex, numCycles);
  	  if (CatchMemoryLeak::s_countAllocated != 0) {
        break; // will catch in error below
  	  }
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  TEST_EQUALITY_CONST(CatchMemoryLeak::s_countAllocated, 0);				// test for valid RCP deletion
}


} // end namespace

#endif // end HAVE_TEUCHOSCORE_CXX11
#endif

