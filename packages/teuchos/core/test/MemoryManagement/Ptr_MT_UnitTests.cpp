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
#ifdef TEUCHOS_DEBUG // this test is only meaningful in DEBUG and would crash in RELEASE with undefined behavior
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

#define UNSET_CYCLE_INDEX -1              // helps to track which thread cycle an event occurred on, -1 means it was never set
struct Cycle_Index_Tracker
{
  Cycle_Index_Tracker()
  {
    danglingReference = UNSET_CYCLE_INDEX;
    scambledMemory = UNSET_CYCLE_INDEX;
    unknownError = UNSET_CYCLE_INDEX;
  }
  int danglingReference;                  // tracks when a dangling reference was hit
  int scambledMemory;                     // tracks when scrambled memory was detected
  int unknownError;                       // tracks an unknown exception - not expected to happen
  std::atomic<int> trackCycle;            // this is for feedback to indicate how many times the thread has actually done a loop
};
  
static void share_ptr_to_threads(Ptr<int> shared_ptr, int theTestValue, Cycle_Index_Tracker & index_tracker) {
  while (!ThreadTestManager::s_bAllowThreadsToRun) {}
  int cycle = 0;
  try {
    while (true) {
      bool bCheckStatus = ThreadTestManager::s_bMainThreadSetToNull;
      
      // there are four possible outcomes of getRawPtr()
         // (1) the ptr debug check returns dangling and a proper throw is detected - in this case we are certain of our result
         // (2) the ptr debug check returns valid and we can read the data (because we are lucky and the data remains valid while we use it)
         // (3) the ptr debug check returns valid, gets deleted by another thread immediately after, but we read the deleted data without knowing because it still contains the proper memory
         // (4) the ptr debug check returns valid, gets deleted by another thread immediately after, is overwriteen by another heap allocation, and we read the scrambled data without knowing
      
      index_tracker.trackCycle = cycle;
      
      if (*shared_ptr != theTestValue) {
        index_tracker.scambledMemory = cycle;
      }
      
      // the scrambler int is trying to jump into the released memory spot through a heap allocation and disrupt the ptr value
      int * pScramblerInt = new int;
      *pScramblerInt = 0; // we hope to set the dangling memory space here
      delete pScramblerInt;
      
      if (bCheckStatus) {
        index_tracker.unknownError = cycle;
        break; // when bCheckStatus true it means we started the loop after the main rcp was set null - we should have thrown a DanglingReference by now
      }
      ++cycle;
    }
  }
  catch(DanglingReferenceError) {
    index_tracker.danglingReference = cycle;
  }
}
  
TEUCHOS_UNIT_TEST( Ptr, mtPtrDangling )
{
  const int numThreads = 4;
  const int theTestValue = 1454083084;
  const int numTests = 1000000;
  
  int countDanglingReferences = 0; // we want to count when it's not trivial (first cycle or last cycle)
  int scrambledMemoryEvents = 0;
  int unknownErrors = 0;
  for (int testCycle = 0; testCycle < numTests; ++testCycle) {
    try {
      int * pInt = new int;                               // create a new int - RCP will own this int and manage its memory
      *pInt = theTestValue;                               // set the int to a test value - we will check for
      RCP<int> shared_rcp = rcp(pInt);                    // first make an RCP
      Ptr<int> shared_ptr = shared_rcp.ptr();             // now make a Ptr which remembers a weak reference to that RCP
    
      std::vector<std::thread> threads;
      ThreadTestManager::s_bAllowThreadsToRun = false;
      ThreadTestManager::s_bMainThreadSetToNull = false;
      Cycle_Index_Tracker index_tracker[numThreads];
      for (int i = 0; i < numThreads; ++i) {
        threads.push_back(std::thread(share_ptr_to_threads, shared_ptr, theTestValue, std::ref(index_tracker[i])));
      }
      ThreadTestManager::s_bAllowThreadsToRun = true;     // let the threads run
      while( index_tracker[0].trackCycle < 1 ) {}         // spin lock the main thread until the sub threads get started with some looping
      shared_rcp = null;                                  // the RCP becomes invalid and the Ptr types all lose their valid object
      ThreadTestManager::s_bMainThreadSetToNull = true;   // tell the threads
      for (unsigned int i = 0; i < threads.size(); ++i) {
        threads[i].join();
      }
      
      // check for danglers
      for (unsigned int i = 0; i < threads.size(); ++i) {
        if (index_tracker[i].danglingReference != -1) {
          ++countDanglingReferences;
        }
        if (index_tracker[i].scambledMemory != -1 ) {
          ++scrambledMemoryEvents;
        }
        if (index_tracker[i].unknownError != -1 ) {
          ++unknownErrors;
        }
      }
    }
    TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
    
    convenience_log_progress(testCycle, numTests);					// this is just output
  }
  
  int expectedDanglingReferences = numThreads * numTests;
  if( countDanglingReferences != expectedDanglingReferences) {
    std::cout << std::endl << "Test FAILED because only " << countDanglingReferences << " dangling references were detected but expected " << expectedDanglingReferences << "." << std::endl;
  }
  else {
    std::cout << "Danglers: " << countDanglingReferences << " Scrambles: " << scrambledMemoryEvents << " ";
  }
  
  if (unknownErrors != 0) {
    std::cout << std::endl << "Detected " << unknownErrors << " dangling references were missed which should have been detected." << std::endl;
  }
  
  TEST_ASSERT(countDanglingReferences  == expectedDanglingReferences) // somewhat arbitrary - verify we detected at least of half of possible danglers
  TEST_EQUALITY_CONST(unknownErrors, 0); // not presently an issue - this is searching for the possibility of a dangling reference missed when it should have been recorded
}
  
} // end namespace

#endif // end HAVE_TEUCHOSCORE_CXX11
#endif // TEUCHOS_DEBUG
