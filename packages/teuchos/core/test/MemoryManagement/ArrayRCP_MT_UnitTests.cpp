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
#include "General_MT_UnitTests.hpp"

#ifdef HAVE_TEUCHOSCORE_CXX11

#include "Teuchos_ArrayRCP.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include <vector>
#include <thread>

namespace {

using Teuchos::ArrayRCP;
using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::null;
using Teuchos::DanglingReferenceError;
  
static void read_arrayrcp_in_thread(ArrayRCP<int> shared_arrayrcp, int expectedValue, std::atomic<int> & countErrors) {
  while (!ThreadTestManager::s_bAllowThreadsToRun) {}
  for( int n = 0; n < 1000; ++n) {
    for (ArrayRCP<int>::const_iterator iter = shared_arrayrcp.begin(); iter < shared_arrayrcp.end(); ++iter) {
      int readAValue = shared_arrayrcp[0];
      if (readAValue != expectedValue) {
        ++countErrors;
      }
    }
  }
}
  
TEUCHOS_UNIT_TEST( ArrayRCP, mtArrayRCPMultipleReads )
{
  const int numThreads = 4;
  const int numTests = 1000;
  const int setValue = 67359487; // arbitrary
  const int arraySize = 10;
  
  std::atomic<int> countErrors(0);
  
  try {
    for (int testCycle = 0; testCycle < numTests; ++testCycle) {
      std::vector<std::thread> threads;
      ThreadTestManager::s_bAllowThreadsToRun = false;
      ArrayRCP<int> shared_arrayrcp(arraySize, setValue); // some array
        
      for (int i = 0; i < numThreads; ++i) {
        threads.push_back( std::thread(read_arrayrcp_in_thread, shared_arrayrcp, setValue, std::ref(countErrors)));
      }

      ThreadTestManager::s_bAllowThreadsToRun = true;     // let the threads run
      for (unsigned int i = 0; i < threads.size(); ++i) {
        threads[i].join();
      }
      convenience_log_progress(testCycle, numTests);					// this is just output
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
}
  
static void read_rcp_of_arrayrcp_in_thread(RCP<ArrayRCP<int>> shared_rcp_of_arrayrcp, int expectedValue, std::atomic<int> & countErrors) {
  while (!ThreadTestManager::s_bAllowThreadsToRun) {}
  for( int n = 0; n < 1000; ++n) {
    for (ArrayRCP<int>::const_iterator iter = shared_rcp_of_arrayrcp->begin(); iter < shared_rcp_of_arrayrcp->end(); ++iter) {
      int readAValue = (*shared_rcp_of_arrayrcp)[0];
      if (readAValue != expectedValue) {
        ++countErrors;
      }
    }
  }
}

TEUCHOS_UNIT_TEST( ArrayRCP, mtRCPofArrayRCPMultipleReads )
{
  const int numThreads = 4;
  const int numTests = 1000;
  const int setValue = 67359487; // arbitrary
  const int arraySize = 10;
  
  std::atomic<int> countErrors(0);
  
  try {
    for (int testCycle = 0; testCycle < numTests; ++testCycle) {
      std::vector<std::thread> threads;
      ThreadTestManager::s_bAllowThreadsToRun = false;
      RCP<ArrayRCP<int>> shared_rcp_of_arrayrcp = rcp(new ArrayRCP<int>(arraySize, setValue)); // some array
        
      for (int i = 0; i < numThreads; ++i) {
        threads.push_back( std::thread(read_rcp_of_arrayrcp_in_thread, shared_rcp_of_arrayrcp, setValue, std::ref(countErrors)) );
      }
        
      ThreadTestManager::s_bAllowThreadsToRun = true;     // let the threads run

      for (unsigned int i = 0; i < threads.size(); ++i) {
        threads[i].join();
      }
      convenience_log_progress(testCycle, numTests);					// this is just output
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
  TEST_EQUALITY(countErrors, 0 );
}

} // end namespace

#endif // end HAVE_TEUCHOSCORE_CXX11
