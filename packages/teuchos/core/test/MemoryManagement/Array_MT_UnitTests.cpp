/*
 * Array_MT_UnitTests.cpp
 *
 *  Created on: May 6, 2016
 *      Author: micheldemessieres
 */

#include "TeuchosCore_ConfigDefs.hpp"

#ifdef HAVE_TEUCHOSCORE_CXX11

//#define REMOVE_MUTEX_LOCK_FOR_ARRAY // adding this line will remove the mutex lock in Array and cause the mtArrayMultipleReads unit test to fail

#include "General_MT_UnitTests.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include <vector>
#include <thread>

namespace {

  using Teuchos::Array;
  using Teuchos::RCP;
  using Teuchos::DanglingReferenceError;
  using Teuchos::RangeError;

  // non-const form
  static void share_nonconst_array_to_threads(RCP<Array<int>> shared_array) {
    while (!ThreadTestManager::s_bAllowThreadsToRun) {}
    for (Array<int>::iterator iter = shared_array->begin(); iter < shared_array->end(); ++iter) {
      // do nothing here - the point is to call the begin()
    }
  }
  
  // const form
  static void share_const_array_to_threads(RCP<const Array<int>> shared_array) {
    while (!ThreadTestManager::s_bAllowThreadsToRun) {}
    for (Array<int>::const_iterator iter = shared_array->begin(); iter < shared_array->end(); ++iter) {
    }
  }
  
  TEUCHOS_UNIT_TEST( Array, mtArrayMultipleReads_NonConst )
  {
    // the point of this test was to validate that multiple threads can safely read an Array
    // we expected it to fail originally because the begin() call in Debug will set extern_arcp_
    // so the first strategy was to make a race condition on that allocation to demonstrate we could see this problem
    // note that begin() is a const but the internal extern_arcp_ object is mutable - that is our target here

    const int numThreads = 4;
    const int numTests = 100;

    for (int testCycle = 0; testCycle < numTests; ++testCycle) {
      try {
        std::vector<std::thread> threads;
        ThreadTestManager::s_bAllowThreadsToRun = false;
        
        RCP<Array<int>> array_rcp = rcp(new Array<int>( 10, 3 )); // makes an array of length 1000 with each element set to 3
        array_rcp->resize( 10, 5 ); // resize the array - need to investigate if there is any subtle issue why we should check this
        
        for (int i = 0; i < numThreads; ++i) {
          threads.push_back( std::thread(share_nonconst_array_to_threads, array_rcp) );
        }
        
        ThreadTestManager::s_bAllowThreadsToRun = true;     // let the threads run
        for (unsigned int i = 0; i < threads.size(); ++i) {
          threads[i].join();
        }
      }
      TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
      
      convenience_log_progress(testCycle, numTests);					// this is just output
    }

    TEST_EQUALITY_CONST(0, 0);
  }

  TEUCHOS_UNIT_TEST( Array, mtArrayMultipleReads_Const )
  {
    // same as prior except now we consider a const Array<int>
    // note that this a much more subtle pipeline for begin() which has to progress from calling the const begin() to non-const begin()
    // this is the reason why we have two mutex members - otherwise the second begin() call would lock indefinitely
    
    const int numThreads = 4;
    const int numTests = 100;
    
    for (int testCycle = 0; testCycle < numTests; ++testCycle) {
      try {
        std::vector<std::thread> threads;
        ThreadTestManager::s_bAllowThreadsToRun = false;
        
        RCP<const Array<int>> array_rcp = rcp(new Array<int>( 10, 3 )); // makes an array of length 1000 with each element set to 3
        
        for (int i = 0; i < numThreads; ++i) {
          threads.push_back( std::thread(share_const_array_to_threads, array_rcp) );
        }
        
        ThreadTestManager::s_bAllowThreadsToRun = true;     // let the threads run
        for (unsigned int i = 0; i < threads.size(); ++i) {
          threads[i].join();
        }
      }
      TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
   
      convenience_log_progress(testCycle, numTests);					// this is just output
    }

    TEST_EQUALITY_CONST(0, 0);
  }

  #ifdef TEUCHOS_DEBUG // this test is only meaningful in DEBUG and would crash with undefined behaviors in RELEASE

  static void call_inserts_on_array(RCP<Array<int>> shared_array, int setValue, int maxArraySize, int finishWhenThisThreadCountCompletes) {
    while (!ThreadTestManager::s_bAllowThreadsToRun) {}
    while (true) { // length can be arbitrary and this will abort when the condition is met - but this should be improved for when it fails to not take so long
      const int insertCount = maxArraySize - shared_array->size();
      // insert some values
      for (int n = 0; n < insertCount; ++n) {
        shared_array->push_back(setValue); // insert some ints with value setValue
      }
      // erase some values
      for (int n = 0; n < insertCount; ++n) {
        shared_array->pop_back(); // remove values so it doesn't get too big - keeping array small avoids condition where dangling is missed and then as array grows it becomes harder to trap
      }
      if (ThreadTestManager::s_countCompletedThreads >= finishWhenThisThreadCountCompletes) {
        break;
      }
    }
  }

  static void scramble_memory(int scrambleValue, int finishWhenThisThreadCountCompletes) {
    while (!ThreadTestManager::s_bAllowThreadsToRun) {}
    // the idea here is to try and fill any heap holes with new int allocations - set them to scrambleValue for bad memory detection
    // better to create all first then delete - otherwise we would just fill and empty the same hole
    // however we also may want to run until the target condition ThreadTestManager::s_bObtainedDesiredCondition is set
    // so I made the scrambler run in batches of 100 new then 100 delete
    while (true) {
      #define ARRAY_SCRAMBLE_SIZE 100 // hard coded this as for thread debugging I didn't want to have extra array methods running while investigating the main operations
      int * tempPtrArray[ARRAY_SCRAMBLE_SIZE];
      for (int n = 0; n < ARRAY_SCRAMBLE_SIZE; ++n) {
        int * pInt = new int;
        *pInt = scrambleValue;
        tempPtrArray[n] = pInt;
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
      outOfRangeError = UNSET_CYCLE_INDEX;
      unknownError = UNSET_CYCLE_INDEX;
    }
    int danglingReference;                  // tracks when a dangling reference was hit
    int scambledMemory;                     // tracks when scrambled memory was detected
    int outOfRangeError;                    // tracks when an out of range error was found
    int unknownError;                       // tracks an unknown exception - not expected to happen
  };

  template<class Array_Template_Form>
  static void do_read_operations_on_array(RCP<Array_Template_Form> shared_array, int setValue, int scrambleValue, int numCyclesInThread,
          Cycle_Index_Tracker & index_tracker, int maxArraySize, bool bCheckReadOperations) {
    while (!ThreadTestManager::s_bAllowThreadsToRun) {}
    int cycle = 0;
    try {
      for (cycle = 0; cycle < numCyclesInThread; ++cycle) {

        // read some values and check for scrambles
        // two things can happen - either we get a range error and get a valid exception
        // or perhaps we just straight out read bad memory in which case we won't get an exception but will try to trap that event here and record it
        
        // Activating bCheckReadOperations will trigger std::out_of_range errors (STL), RangeError (Trilinos), and sometimes a detected scramble event meaning we though the data was fine but read the wrong value
        // This all comes from the weak ArrayRCP which currently can't protect against some race conditions
        if (bCheckReadOperations) {
          int readValue = shared_array->at(0);
          if( readValue != setValue ) {
            // was using this to see if things were working properly - but we don't necessarily expect the scrambled int to always propagate here - something else could be going on
            if( readValue == scrambleValue) {
              index_tracker.scambledMemory = cycle;
            }
            else {
              index_tracker.scambledMemory = cycle;
            }
          }
        }
        
        for( Array<int>::const_iterator iter = shared_array->begin(); iter < shared_array->end(); ++iter) {} // do nothing - the dangling will be generated by ++iter
      }
    }
    catch (DanglingReferenceError) {
      index_tracker.danglingReference = cycle; // this test is constructued so we should always hit this - but the nature of the threads means it's not guaranteed to happen in any set time
    }
    catch (RangeError) {
      index_tracker.outOfRangeError = cycle;
      //std::cout << std::endl << "Got RangeError Exception" << std::endl;
    }
    catch (std::out_of_range) {
      index_tracker.outOfRangeError = cycle; // Note that currently I am counting Trilinos RangeError and STL std::out_of_range as all the same - they both happen since the bad memory read can be triggered at any time
      //std::cout << std::endl << "Got std::out_of_range Exception" << std::endl;
    }
    catch (...) {
      std::cout << std::endl << "GOT UNEXPECTED UNHANDLED EXCEPTION" << std::endl;
    }

    ++ThreadTestManager::s_countCompletedThreads; // advertise we are done - when all these threads complete the push/pop thread and scambler thread will quit
  }

  bool runArrayDanglingReferenceTest( bool bUseConstVersion, bool bUseScramblerThread, bool bCheckReadOperations )
  {
    // we create a non const Array<int> and have it do many insert calls while other threads try to read it
    const int numThreads = 4; // note that 0 is the insert thread, and 1 is the scrambler thread - so set this to be 4 or more - 3 is not sufficient as you'll not have multiple threads calling read

    const int numTests = bCheckReadOperations ? 50000 : 50000;
    const int setValue = 1;
    const int numCyclesInThread = 10000; // once we hit the dangle we stop, so this number can/should be large ... if things are not working as we hoped...
    const int scrambleValue = 12345;
    const int maxArraySize = 100;
    
    int countTotalTestRuns = 0;                     // the maximum number of errors we can detect (one per thread which is doing read operations - does not count the push/pop thread or the scramble thread)
    int countDetectedDanglingReferences = 0;        // the actual number of dangling references we detect
    int countScrambledMemoryEvents = 0;             // this counts the times the dangling reference missed (thought memory was fine) but it was actually not and it was found to be overwritten to the scrambleValue
    int countOutOfRangeEvents = 0;                  // this counts out of range errors which are currently triggered by reading the array with a concurrent write - currently Trilinos RangeError and STL std::out_of_range are grouped together


    for (int testCycle = 0; testCycle < numTests; ++testCycle) {
      try {
        std::vector<std::thread> threads;
        ThreadTestManager::s_bAllowThreadsToRun = false;
        ThreadTestManager::s_countCompletedThreads = 0;
        int finishWhenThisThreadCountCompletes = numThreads - ( bUseScramblerThread ? 2 : 1 ); // 0 is pushing/popping, 1 is optional memory reading/writing. The rest are the reader threads looking for troubles
        Cycle_Index_Tracker index_tracker[numThreads];              // I avoid using general Arrays as we are testing them, makes debugging thread issues easier
        
        RCP<Array<int>> array_rcp = rcp(new Array<int>(1, setValue)); // makes an array of length 1 - so we will cycle from size 1 to size maxArraySize then back to 1

        for (int i = 0; i < numThreads; ++i) {
          switch (i)
          {
            case 0:
              threads.push_back( std::thread(call_inserts_on_array, array_rcp, setValue, maxArraySize, finishWhenThisThreadCountCompletes) );
              break;
            case 1:
              threads.push_back( std::thread(scramble_memory, scrambleValue, finishWhenThisThreadCountCompletes) );
              break;
            default:
              ++countTotalTestRuns;
              if (bUseConstVersion) {
                threads.push_back( std::thread(do_read_operations_on_array< const Array<int> >, array_rcp, setValue, scrambleValue, numCyclesInThread, std::ref(index_tracker[i]), maxArraySize, bCheckReadOperations));
              }
              else {
                threads.push_back( std::thread(do_read_operations_on_array< Array<int> >, array_rcp, setValue, scrambleValue, numCyclesInThread, std::ref(index_tracker[i]), maxArraySize, bCheckReadOperations));
              }
              break;
          }
        }

        ThreadTestManager::s_bAllowThreadsToRun = true;     // let the threads run
        for (unsigned int i = 0; i < threads.size(); ++i) {
          threads[i].join();
        }

        // for this test to be legitimate we need dangling reference detection to be happening mid loop - so verify it was found and not on the first or last cycle - though this could happen sometimes
        for (unsigned int i = 0; i < threads.size(); ++i) {
          if (index_tracker[i].danglingReference != UNSET_CYCLE_INDEX ) { // && danglingReferenceDetectionCycle[i] != 0 && danglingReferenceDetectionCycle[i] != numCyclesInThread-1 ) {
            ++countDetectedDanglingReferences;
          }
          if (index_tracker[i].scambledMemory != UNSET_CYCLE_INDEX ) {
            ++countScrambledMemoryEvents;
          }
          if (index_tracker[i].outOfRangeError != UNSET_CYCLE_INDEX ) {
            ++countOutOfRangeEvents;
          }
        }
      }
      catch( ... ) {
        std::cout << "Test threw an exception but did not handle it!" << std::endl;
      }

      convenience_log_progress(testCycle, numTests);					// this is just output
    }

    int totalDetectedErrors = countDetectedDanglingReferences + countOutOfRangeEvents;
    bool bPassed_ProperErrorCount = (totalDetectedErrors == countTotalTestRuns);
    bool bPassed_ScrambledMemoryCount = (countScrambledMemoryEvents == 0);
    bool bPass = bPassed_ProperErrorCount && bPassed_ScrambledMemoryCount;
    if (!bPass) {
      std::cout << std::endl; // cosmetic - get these errors on a new line
    }
    
    if( !bPassed_ProperErrorCount ) {
      std::cout << "Test FAILED because it detected " << countDetectedDanglingReferences << " Dangling References and " << countOutOfRangeEvents << " Out of Range events which accounts for a total of " << totalDetectedErrors << " out of " <<  countTotalTestRuns << ". We missed something! Currently some undefined memory scrambling is expected due to the weak reference." << std::endl;
    }
    else if (!bPassed_ScrambledMemoryCount) {
      std::cout << "Test FAILED because it detected " << countScrambledMemoryEvents << " scrambled memory events. Note this is currently an expected behavior - no fix yet." << std::endl;
    }
    else if (countDetectedDanglingReferences == totalDetectedErrors) {
      std::cout << "Detected " << totalDetectedErrors << " Danglers "; // might suppress this later - this is a pass message
    }
    else { // in this case we had a mix of danglers and range errors
      std::cout << "Detected " << countDetectedDanglingReferences << "-" << countOutOfRangeEvents << " Danglers-RangeErrors "; // might suppress this later - this is a pass message
    }

    return bPass;
  }
  
  TEUCHOS_UNIT_TEST( Array, mtArrayDanglingReference_NonConst )
  {
    bool bPass = runArrayDanglingReferenceTest( false, true, false );
    TEST_ASSERT( bPass ) // in our current form we demand 100% detection
  }
  
  TEUCHOS_UNIT_TEST( Array, mtArrayDanglingReference_Const )
  {
    bool bPass = runArrayDanglingReferenceTest( true, true, false );
    TEST_ASSERT( bPass ) // in our current form we demand 100% detection
  }
  
  /*
  TEUCHOS_UNIT_TEST( Array, mtArrayDanglingReference_NonConst_ReadValues )
  {
    bool bPass = runArrayDanglingReferenceTest( false, true, true );
    TEST_ASSERT( bPass ) // in our current form we demand 100% detection
  }
  
  TEUCHOS_UNIT_TEST( Array, mtArrayDanglingReference_Const_ReadValues )
  {
    bool bPass = runArrayDanglingReferenceTest( true, true, true );
    TEST_ASSERT( bPass ) // in our current form we demand 100% detection
  }
  */
#endif // TEUCHOS_DEBUG
  
} // end namespace

#endif // HAVE_TEUCHOSCORE_CXX11



