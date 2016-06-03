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
#include "General_MT_UnitTests.hpp"
#include "Teuchos_Ptr.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include <vector>
#include <thread>
#include <atomic>

namespace {

using Teuchos::Ptr;
using Teuchos::RCP;
using Teuchos::DanglingReferenceError;
using Teuchos::null;
using Teuchos::rcp;
using Teuchos::ptrFromRef;
using Teuchos::rcpFromPtr;
  
/*

 Notes on Ptr and RCP
 
 When we construct a Ptr from an RCP, we will store a weak reference to the RCP with the Ptr
 This happens in the RCP::ptr() function which calls create_weak()
 
 In our thread test, we make an RCP<int> and then create a ptr from the RCP, all in the main thread
 Then we pass the Ptr to all the threads - the idea is that they would all be using this for some non persisting operation
 
 When the threads are halfway done with their operations we will set the RCP to null in the main thread
 
 Now the Ptr in the threads should properly detect a DanglingReference error
 This would be a programming error - we set up the threads to have access to the Ptr but then killed the source object improperly
 
 Ptr will call getRawPtr() which will do a debug_assert_valid_ptr()
 This assert is essentially NOT thread safe if called on an RCP Weak (which is what we are doing)
 The assert can pass and then the main thread will kill the RCP object
 Then the shared_ptr will return it's ptr_ member which is now pointing to junk
 
 So our first goal is to properly show that the test can find this problem
 We are going to try and use that ptr
 It's not a legitimate use of Ptr to have multiple threads manipulating that memoery
 However it should be legitimate use to have multiple threads read that memory - they should either get the right int value or a proper dangling reference error
 They should not improperly read bad memory without knowing about it.
 
 What we want to establish is that the threads can still properly detect the dangling reference error

 Note that the real issue here is probably going to be how to restrict the access
 A Ptr from an RCP is always going to have weak RCP information about the status
 That means that we really can't do anything with ptr, ever, if it's possible the source RCP can be going to null
 I think the best we can do here is make sure that the debug check will properly execute on the dangling reference check
 By design the program should not have allowed the RCP to go away while the threads are in progress
 
 Even if we make the weak rcp have an atomic safe debug check, we still can't guarantee the ptr is valid after the check is done.
 However we can consider a pattern when Ptr can lock (convert it's weak RCP to a strong RCP) and then safely work.
*/

static void share_ptr_to_threads(Ptr<int> shared_ptr, int theTestValue, int totalCycles, int & danglingReferenceDetectionCycle, int & scrambledMemoryDetectionCycle, std::atomic<int> & trackCycle) {
  while (!ThreadTestManager::s_bAllowThreadsToRun) {}
  int cycle = 0;
  try {
    for (cycle = 0; cycle < totalCycles; ++cycle) {
      // there are three possible outcomes of getRawPtr()
         // (1) the ptr is dangling and a proper throw is detected
         // (2) the ptr is valid and we can read the data
         // (3) the ptr is incorrectly read as valid, becomes invalid, and we read scrambled data without knowing it - this is what we want to avoid
      
      int * pGetPossibleBadPtr = shared_ptr.getRawPtr(); // this may return a ptr which is now pointing to released memory but there is a good chance the memory won't be rewritten yet
      
      // the scrambler int is trying to jump into the released memory spot through a heap allocation and disrupt the ptr value
      int * pScramblerInt = new int;
      *pScramblerInt = 0;
      delete pScramblerInt;
      
      trackCycle = cycle;
      
      if (*pGetPossibleBadPtr != theTestValue) {
        scrambledMemoryDetectionCycle = cycle;
      }
    }
  }
  catch(DanglingReferenceError) {
    danglingReferenceDetectionCycle = cycle;
  }
}
  
TEUCHOS_UNIT_TEST( Ptr, mtPtrUsedInThreads )
{
  const int numThreads = 4;
  const int numCyclesInThread = 100;
  const int theTestValue = 1454083084;
  const int numTests = 100000;
  
  int countNonTrivialDanglingReferences = 0; // we want to count when it's not trivial (first cycle or last cycle)
  int approximateNumberOfDanglingReferenceDetections = numThreads * numTests; // each thread is one possible dangling reference detection but we don't necessarily always read one - most of the time
  int scrambledMemoryEvents = 0;
  for (int testCycle = 0; testCycle < numTests; ++testCycle) {
    try {
      int * pInt = new int;                                 // create a new int - RCP will own this int and manage its memory
      *pInt = theTestValue;                                 // set the int to a test value - we will check for
      RCP<int> shared_rcp = rcp(pInt);                    // first make an RCP
      Ptr<int> shared_ptr = shared_rcp.ptr();             // now make a Ptr which remembers a weak reference to that RCP
    
      std::vector<std::thread> threads;
      ThreadTestManager::s_bAllowThreadsToRun = false;
      int danglingReferenceDetectionCycle[numThreads];
      int scrambledMemoryDetectionCycle[numThreads];
      std::atomic<int> trackCycle[numThreads];            // make these atomic - we will kill the rcp mid loop
      for (int i = 0; i < numThreads; ++i) {
        trackCycle[i] = -1; // unset
        danglingReferenceDetectionCycle[i] = -1; // unset
        scrambledMemoryDetectionCycle[i] = -1; // unset
        threads.push_back(std::thread(share_ptr_to_threads, shared_ptr, theTestValue, numCyclesInThread, std::ref(danglingReferenceDetectionCycle[i]), std::ref(scrambledMemoryDetectionCycle[i]), std::ref(trackCycle[i])));
      }
      ThreadTestManager::s_bAllowThreadsToRun = true;     // let the threads run
      while( trackCycle[0] < numCyclesInThread/2 ) {}     // spin lock the main thread until the sub threads are getting up to about halfway done
      shared_rcp = null;                                  // the RCP becomes invalid and the Ptr types all lose their valid object
      for (int i = 0; i < threads.size(); ++i) {
        threads[i].join();
      }
      
      // for this test to be legitimate we need dangling reference detection to be happening mid loop - so verify it was found and not on the first or last cycle - though this could happen sometimes
      for (int i = 0; i < threads.size(); ++i) {
        if (danglingReferenceDetectionCycle[i] != -1 && danglingReferenceDetectionCycle[i] != 0 && danglingReferenceDetectionCycle[i] != numCyclesInThread-1 ) {
          ++countNonTrivialDanglingReferences;
        }
        if (scrambledMemoryDetectionCycle[i] != -1 ) {
          ++scrambledMemoryEvents;
        }
      }
    }
    TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  }
  
  int requiredDanglingReferenceCount = approximateNumberOfDanglingReferenceDetections/2;
  bool bDanglingReferenceDetectionCountIsOK = (countNonTrivialDanglingReferences > requiredDanglingReferenceCount);
  
  if( !bDanglingReferenceDetectionCountIsOK ) {
    std::cout << "Test completed with " << countNonTrivialDanglingReferences << " non-trivial dangling reference detections which is below a desired threshold of ." << requiredDanglingReferenceCount << std::endl;
  }
  
  if (scrambledMemoryEvents != 0) {
    std::cout << "Test failed because it detected " << scrambledMemoryEvents << " scrambled memory events. Note this is currently an expected behavior - no fix yet." << std::endl;
  }
  
  TEST_ASSERT( bDanglingReferenceDetectionCountIsOK ) // somewhat arbitrary - verify we detected at least of half of possible danglers
  TEST_EQUALITY_CONST(scrambledMemoryEvents, 0); // this is ultimately the final test - when we fix this issue we should never accidentally read bad memory
}

} // end namespace

#endif // end HAVE_TEUCHOSCORE_CXX11
