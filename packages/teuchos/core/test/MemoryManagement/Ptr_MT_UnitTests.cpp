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

// This will be activated later - currently it detects undefined memory events and need to resolve how we will approach this
#ifdef CURRENTLY_DISABLED // HAVE_TEUCHOSCORE_CXX11
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

#ifdef TEUCHOS_DEBUG // this test is only meaningful in DEBUG and would crash in RELEASE with undefined behavior
  
static void share_ptr_to_threads(Ptr<int> shared_ptr, int theTestValue, int totalCycles, int & danglingReferenceDetectionCycle, int & scrambledMemoryDetectionCycle, std::atomic<int> & trackCycle) {
  while (!ThreadTestManager::s_bAllowThreadsToRun) {}
  int cycle = 0;
  try {
    for (cycle = 0; cycle < totalCycles; ++cycle) {
      // there are four possible outcomes of getRawPtr()
         // (1) the ptr debug check returns dangling and a proper throw is detected - in this case we are certain of our result
         // (2) the ptr debug check returns valid and we can read the data (because we are lucky and the data remains valid while we use it)
         // (3) the ptr debug check returns valid, gets deleted by another thread immediately after, but we read the deleted data without knowing because it still contains the proper memory
         // (4) the ptr debug check returns valid, gets deleted by another thread immediately after, is overwriteen by another heap allocation, and we read the scrambled data without knowing
      
      trackCycle = cycle;
      
      if (*shared_ptr != theTestValue) {
        scrambledMemoryDetectionCycle = cycle;
      }
      
      // the scrambler int is trying to jump into the released memory spot through a heap allocation and disrupt the ptr value
      int * pScramblerInt = new int;
      *pScramblerInt = 0; // we hope to set the dangling memory space here
      delete pScramblerInt;
    }
  }
  catch(DanglingReferenceError) {
    danglingReferenceDetectionCycle = cycle;
  }
}
  
TEUCHOS_UNIT_TEST( Ptr, mtPtrDangling )
{
  const int numThreads = 4;
  const int numCyclesInThread = 100;
  const int theTestValue = 1454083084;
  const int numTests = 1000000;
  
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
      for (unsigned int i = 0; i < threads.size(); ++i) {
        threads[i].join();
      }
      
      // for this test to be legitimate we need dangling reference detection to be happening mid loop - so verify it was found and not on the first or last cycle - though this could happen sometimes
      for (unsigned int i = 0; i < threads.size(); ++i) {
        if (danglingReferenceDetectionCycle[i] != -1 && danglingReferenceDetectionCycle[i] != 0 && danglingReferenceDetectionCycle[i] != numCyclesInThread-1 ) {
          ++countNonTrivialDanglingReferences;
        }
        if (scrambledMemoryDetectionCycle[i] != -1 ) {
          ++scrambledMemoryEvents;
        }
      }
    }
    TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
    
    convenience_log_progress(testCycle, numTests);					// this is just output
  }
  
  int requiredDanglingReferenceCount = approximateNumberOfDanglingReferenceDetections/2;
  bool bDanglingReferenceDetectionCountIsOK = (countNonTrivialDanglingReferences > requiredDanglingReferenceCount);
  
  if( !bDanglingReferenceDetectionCountIsOK ) {
    std::cout << std::endl << "Test completed with " << countNonTrivialDanglingReferences << " non-trivial dangling reference detections which is below a desired threshold of " << requiredDanglingReferenceCount << "." << std::endl;
  }
  
  if (scrambledMemoryEvents != 0) {
    std::cout << std::endl << "Test failed because it detected " << scrambledMemoryEvents << " scrambled memory events. Note this is currently an expected behavior - no fix yet." << std::endl;
  }
  else {
    std::cout << std::endl << "No " << scrambledMemoryEvents << " events detected. In the current code setup we expect these to happen." << std::endl;
  }
  
  TEST_ASSERT( bDanglingReferenceDetectionCountIsOK ) // somewhat arbitrary - verify we detected at least of half of possible danglers
  TEST_EQUALITY_CONST(scrambledMemoryEvents, 0); // this is ultimately the final test - when we fix this issue we should never accidentally read bad memory
}
#endif // TEUCHOS_DEBUG
  
} // end namespace

#endif // end HAVE_TEUCHOSCORE_CXX11
