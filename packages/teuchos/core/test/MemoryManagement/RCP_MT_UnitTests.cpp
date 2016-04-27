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
#include "Teuchos_RCP.hpp"
#include "Teuchos_getConst.hpp"
#include "Teuchos_getBaseObjVoidPtr.hpp"
#ifdef HAVE_TEUCHOSCORE_CXX11
#  include "Teuchos_RCPStdSharedPtrConversions.hpp"
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

static void make_large_number_of_copies(RCP<int> ptr) {
  std::vector<RCP<int> > ptrs(10000, ptr);
}

TEUCHOS_UNIT_TEST( RCP, mtRefCount )
{
  RCP<int> ptr(new int);
  TEST_ASSERT(ptr.total_count() == 1);
  std::vector<std::thread> threads;
  for (int i = 0; i < 4; ++i) {
    threads.push_back(std::thread(make_large_number_of_copies, ptr));
  }
  for (int i = 0; i < 4; ++i) {
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
	std::cout << "Running in Debug Mode. Multi-Threading originally caused the node tracing to fail in Debug. Let's check now." << std::endl;
  #else
	std::cout << "Running in Release Mode. This test was always fine in Multi-Threading." << std::endl;
  #endif

  std::vector<std::thread> threads;
  int numThreads = 4;
  for (int i = 0; i < numThreads; ++i) {
    threads.push_back(std::thread(create_independent_rcp_objects));
  }
  for (int i = 0; i < threads.size(); ++i) {
    threads[i].join();
  }
}

} // namespace
