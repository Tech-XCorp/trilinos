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
    for (ArrayView<int>::iterator iter = shared_arrayview->begin(); iter < shared_arrayview->end(); ++iter) {
      // do nothing here - the point is to call the begin()
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
    
    TEST_EQUALITY_CONST(0, 0);
  }
  
  #ifdef TEUCHOS_DEBUG // this test is only meaningful in DEBUG and would crash in RELEASE with undefined behaviors
  
  // note this test mirrors the Ptr test - the mechanisms we are considering here are essentially equivalent
  // when a weak RCP is raced, it can indicate that it is not dangling, and then be read on junk memory because the delete happens right after the dangling check
  static void share_arrayview_to_threads(ArrayView<int> shared_arrayview, int theTestValue, int totalCycles, int & danglingReferenceDetectionCycle, int & scrambledMemoryDetectionCycle, std::atomic<int> & trackCycle) {
    while (!ThreadTestManager::s_bAllowThreadsToRun) {}
    int cycle = 0;
    try {
      for (cycle = 0; cycle < totalCycles; ++cycle) {
        int tryToReadAValue = shared_arrayview[0];    // this may return junk data if the new heap allocation jumped in - any of the member values could be junk
        trackCycle = cycle;
        if (tryToReadAValue != theTestValue) {
          scrambledMemoryDetectionCycle = cycle;
        }
        
        // here try to scramble the dangling reference memory space - note that I am using an identical allocation as the original in the main test which is a vector<int> of size 10.
        // hoping that can increase the chance to claim the dangling memory space
        std::vector<int> * pScramblerArray = new std::vector<int>(10, 0); // may fill the dangling memory space with 0 values
        delete pScramblerArray;
      }
    }
    catch (DanglingReferenceError) {
      danglingReferenceDetectionCycle = cycle;
    }
    catch (RangeError) {
      std::cout << std::endl << "Range Error!" << std::endl;
    }
    catch (std::out_of_range) {
      std::cout << std::endl << "std::out_of_range!" << std::endl;
    }
    catch (...) {
      std::cout << std::endl << "Unknown and unhandled exception!" << std::endl;
    }
  }
  
  // This test sort of working but probably should wait to finalize until determining issues for Ptr which considers a similar pattern
  // We are checking for dangling reference detection when the root thread sets a shared ArrayRCP (viewed through an ArrayView) to null
  // Eventually we should detect scramble events on this test
  TEUCHOS_UNIT_TEST( ArrayView, mtArrayViewDangling )
  {
    const int numThreads = 4;
    const int numCyclesInThread = 1000000; // test ends when it hits the dangling - still have some off behaviors where a thread will shut off for a bit and that makes this not very robust - the large number guarantees we hit the danglers, I hope...
    const int theTestValue = 1454083084;
    const int numTests = 25000;
    
    int countDanglingReferences = 0; // we want to count when it's not trivial (first cycle or last cycle)
    int scrambledMemoryEvents = 0;
    for (int testCycle = 0; testCycle < numTests; ++testCycle) {
      try {
        ArrayRCP<int> arrayrcp = arcp(rcp(new std::vector<int>(10, theTestValue)));   // first make an arrayrcp which we will kill later
        ArrayView<int> shared_arrayview = arrayrcp();         // now make an ArrayView which has a reference to the arrayrcp

        std::vector<std::thread> threads;
        ThreadTestManager::s_bAllowThreadsToRun = false;
        int danglingReferenceDetectionCycle[numThreads];
        int scrambledMemoryDetectionCycle[numThreads];
        std::atomic<int> trackCycle[numThreads];            // make these atomic - we will kill the arrayrcp mid loop
        for (int i = 0; i < numThreads; ++i) {
          trackCycle[i] = -1; // unset
          danglingReferenceDetectionCycle[i] = -1; // unset
          scrambledMemoryDetectionCycle[i] = -1; // unset
          threads.push_back(std::thread(share_arrayview_to_threads, shared_arrayview, theTestValue, numCyclesInThread, std::ref(danglingReferenceDetectionCycle[i]), std::ref(scrambledMemoryDetectionCycle[i]), std::ref(trackCycle[i])));
        }
        ThreadTestManager::s_bAllowThreadsToRun = true;     // let the threads run
        
        // spin lock the main thread until the sub threads have run a few cycles
        const int minimumCyclesBeforeRCPGoesNull = 10;
        if (minimumCyclesBeforeRCPGoesNull < numCyclesInThread) { // for small number of cycles just skip this - though normally we would not do this
          bool bThreadsRanFarEnough = false;
          while( !bThreadsRanFarEnough )
          {
            bThreadsRanFarEnough = true;
            for (int i = 0; i < numThreads; ++i) {
              if (trackCycle[i] < minimumCyclesBeforeRCPGoesNull) {
                bThreadsRanFarEnough = false; // keep waiting
              }
            }
          }
        }
        
        arrayrcp = null;                                    // the ArrayRCP becomes invalid and the ArrayView types all lose their valid object
        
        for (unsigned int i = 0; i < threads.size(); ++i) {
          threads[i].join();
        }
        
        // for this test to be legitimate we need dangling reference detection to be happening mid loop - so verify it was found and not on the first or last cycle - though this could happen sometimes
        for (unsigned int i = 0; i < threads.size(); ++i) {
          if (danglingReferenceDetectionCycle[i] != -1) {
            ++countDanglingReferences;
          }
          if (scrambledMemoryDetectionCycle[i] != -1 ) {
            ++scrambledMemoryEvents;
          }
        }
      }
      TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
      
      convenience_log_progress(testCycle, numTests);					// this is just output
    }
    
    int requiredDanglingReferenceCount = numThreads * numTests;
    bool bDanglingReferenceDetectionCountIsOK = (countDanglingReferences == requiredDanglingReferenceCount);
    
    if( !bDanglingReferenceDetectionCountIsOK ) {
      std::cout << std::endl << "Detected " << countDanglingReferences << " Dangling References. Expected " << requiredDanglingReferenceCount << "." << std::endl;
    }
    
    if (scrambledMemoryEvents != 0) {
      std::cout << std::endl << "Test failed because it detected " << scrambledMemoryEvents << " scrambled memory events. Note this is currently an expected behavior - no fix yet." << std::endl;
    }
    
    std::cout << "Counted " << countDanglingReferences << " Danglers ";
  
    TEST_ASSERT( bDanglingReferenceDetectionCountIsOK )
    TEST_EQUALITY_CONST(scrambledMemoryEvents, 0); // this is ultimately the final test - when we fix this issue we should never accidentally read bad memory
  }
  #endif // TEUCHOS_DEBUG
  
} // end namespace

#endif // HAVE_TEUCHOSCORE_CXX11