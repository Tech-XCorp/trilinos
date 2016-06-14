/*
 * ArrayView_MT_UnitTests.cpp
 *
 *  Created on: May 6, 2016
 *      Author: micheldemessieres
 */

#include "TeuchosCore_ConfigDefs.hpp"

#ifdef HAVE_TEUCHOSCORE_CXX11

#include "General_MT_UnitTests.hpp"
#include "Teuchos_ArrayView.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include <vector>
#include <thread>

namespace {

using Teuchos::ArrayView;
using Teuchos::null;
using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::Array;
using Teuchos::ArrayRCP;
using Teuchos::RangeError;
using Teuchos::DanglingReferenceError;

static void read_arrayview_in_thread(RCP<ArrayView<int>> shared_arrayview) {
  while (!ThreadTestManager::s_bAllowThreadsToRun) {}
  for( int n = 0; n < 1000; ++n) {
    for (ArrayView<int>::iterator iter = shared_arrayview->begin(); iter < shared_arrayview->end(); ++iter) {
      // do nothing here - the point is to call the iterators
    }
  }
}

// sanity check - this mirrors the Array test (which will fail without mutex protection) but this is ok because the ArrayView begin does not have any mutable behavior (it is true const)
TEUCHOS_UNIT_TEST( ArrayView, mtArrayViewMultipleReads )
{
  const int numThreads = 4;
  const int numTests = 1000;
  for (int testCycle = 0; testCycle < numTests; ++testCycle) {
    try {
      std::vector<std::thread> threads;
      ThreadTestManager::s_bAllowThreadsToRun = false;
      Array<int> array(10, 3); // some array
      RCP<ArrayView<int>> arrayview_rcp = rcp(new ArrayView<int>(array));

      for (int i = 0; i < numThreads; ++i) {
        threads.push_back( std::thread(read_arrayview_in_thread, arrayview_rcp) );
      }

      ThreadTestManager::s_bAllowThreadsToRun = true;     // let the threads run
      for (unsigned int i = 0; i < threads.size(); ++i) {
        threads[i].join();
      }
    }
    TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);

    convenience_log_progress(testCycle, numTests);					// this is just output
  }

  TEST_EQUALITY_CONST(0, 0); // right now this test is just looking for trouble - so we don't actually have a verification - a failed test would be corrupted memory for example
}

#ifdef TEUCHOS_DEBUG // this test is only meaningful in DEBUG and would crash in RELEASE with undefined behaviors

static void scramble_memory(int scrambleValue, int testArraySize, int finishWhenThisThreadCountCompletes) {
  while (!ThreadTestManager::s_bAllowThreadsToRun) {}
  // the idea here is to try and fill any heap holes with new int allocations
  while (true) {
    #define ARRAY_SCRAMBLE_SIZE 100 // hard coded this as for thread debugging I didn't want to have extra array methods running while investigating the main operations
    std::vector<int> * tempPtrArray[ARRAY_SCRAMBLE_SIZE];
    for (int n = 0; n < ARRAY_SCRAMBLE_SIZE; ++n) {
      tempPtrArray[n] = new std::vector<int>(testArraySize, scrambleValue); // if the scramble thread does not allocate std::vector chunks identical to the main thread it won't have any chance to trigger the scrambled events
    }
    for (int n = 0; n < ARRAY_SCRAMBLE_SIZE; ++n) {
      delete tempPtrArray[n];
    }
    if (ThreadTestManager::s_countCompletedThreads >= finishWhenThisThreadCountCompletes) {
      break;
    }
  }
}

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
  std::atomic<int> trackCycle;                         // this is for feedback to indicate how many times the thread has actually done a loop
};

// note this test mirrors the Ptr test - the mechanisms we are considering here are essentially equivalent
// when a weak RCP is raced, it can indicate that it is not dangling, and then be read on junk memory because the delete happens right after the dangling check
static void share_arrayview_to_threads(ArrayView<int> shared_arrayview, int theTestValue, Cycle_Index_Tracker & index_tracker) {
  while (!ThreadTestManager::s_bAllowThreadsToRun) {}
  int cycle = 0;
  try {
    while (true) {
      bool bCheckStatus = ThreadTestManager::s_bMainThreadSetToNull;
      int tryToReadAValue = shared_arrayview[0];    // this may return junk data if the new heap allocation jumped in - any of the member values could be junk
      index_tracker.trackCycle = cycle;
      if (tryToReadAValue != theTestValue) {
        index_tracker.scambledMemory = cycle; // if we get here we had an ok from the dangling reference check, but then memory was deleted and reallocated to a new value by the scramble thread - a rare but possible condition
      }

      if (bCheckStatus) {
        index_tracker.unknownError = cycle;
        break; // when bCheckStatus true it means we started the loop after the main rcp was set null - we should have thrown a DanglingReference by now
      }
      ++cycle;
    }
  }
  catch (DanglingReferenceError) {
    index_tracker.danglingReference = cycle; // loop ends - we got the dangling reference
  }
  catch (...) {
    std::cout << std::endl << "Unknown and unhandled exception!" << std::endl;
  }

  ++ThreadTestManager::s_countCompletedThreads;
}

// This test closely mirrors the Ptr test - we create an ArrayView from an ArrayRCP std::vector, we share it, then we kill the original ArrayRCP
TEUCHOS_UNIT_TEST( ArrayView, mtArrayViewDangling )
{
  const int numThreads = 3;
  const int theTestValue = 6635786587; // some value
  const int numTests = 50000;
  const int scrambleValue = 57252789578; // some other value
  const int testArraySize = 3;

  int countDanglingReferences = 0; // we want to count when it's not trivial (first cycle or last cycle)
  int scrambledMemoryEvents = 0;
  int unknownErrors = 0;
  int finishWhenThisThreadCountCompletes = numThreads - 1; // 0 is the scrambling thread doing constant new/delete. The rest are the reader threads looking for troubles
  for (int testCycle = 0; testCycle < numTests; ++testCycle) {
    try {
      ThreadTestManager::s_countCompletedThreads = 0;

      ArrayRCP<int> arrayrcp = arcp(rcp(new std::vector<int>(testArraySize, theTestValue)));   // first make an arrayrcp which we will kill later
      ArrayView<int> shared_arrayview = arrayrcp();         // now make an ArrayView which has a reference to the arrayrcp

      std::vector<std::thread> threads;
      ThreadTestManager::s_bAllowThreadsToRun = false;
      ThreadTestManager::s_bMainThreadSetToNull = false;
      Cycle_Index_Tracker index_tracker[numThreads];
      for (int i = 0; i < numThreads; ++i) {
        switch(i) {
          case 0:
          {
            threads.push_back(std::thread(scramble_memory, scrambleValue, testArraySize, finishWhenThisThreadCountCompletes));
          }
          break;
          default:
          {
            threads.push_back(std::thread(share_arrayview_to_threads, shared_arrayview, theTestValue, std::ref(index_tracker[i])));
          }
          break;
        }
      }
      ThreadTestManager::s_bAllowThreadsToRun = true;     // let the threads start running

      while (index_tracker[1].trackCycle < 1) {
        // spin lock until we have confirmed the sub threads did something
      }

      arrayrcp = null;                                    // the ArrayRCP becomes invalid and the ArrayView types all lose their valid object - now we start getting dangling references
      ThreadTestManager::s_bMainThreadSetToNull = true;   // tell the threads

      for (unsigned int i = 0; i < threads.size(); ++i) {
        threads[i].join();
      }

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

  int requiredDanglingReferenceCount = (numThreads-1) * numTests;
  bool bDanglingReferenceDetectionCountIsOK = (countDanglingReferences == requiredDanglingReferenceCount);

  if( !bDanglingReferenceDetectionCountIsOK ) {
    std::cout << std::endl << "Detected " << countDanglingReferences << " Dangling References but should have found " << requiredDanglingReferenceCount << "." << std::endl;
  }
  else {
    std::cout << "Danglers: " << countDanglingReferences << " Scrambles: " << scrambledMemoryEvents << " ";
  }

  if (unknownErrors != 0) {
    std::cout << std::endl << "Detected " << unknownErrors << " dangling references were missed which should have been detected." << std::endl;
  }

  TEST_ASSERT( bDanglingReferenceDetectionCountIsOK )
  TEST_EQUALITY_CONST(unknownErrors, 0); // this is ultimately the final test - when we fix this issue we should never accidentally read bad memory
}

#endif // TEUCHOS_DEBUG

} // end namespace

#endif // HAVE_TEUCHOSCORE_CXX11



