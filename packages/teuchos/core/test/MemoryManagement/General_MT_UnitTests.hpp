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

#ifndef TEUCHOS_GENERAL_MT_UNITTESTS_HPP
#define TEUCHOS_GENERAL_MT_UNITTESTS_HPP

#include "TeuchosCore_ConfigDefs.hpp"

#ifdef HAVE_TEUCHOSCORE_CXX11 // all MT unit testing requires HAVE_TEUCHOSCORE_CXX11 which is defined through TeuchosCore_ConfigDefs.hpp

#include <iostream>

namespace {

// this is a convenience function for outputting percent complete information for long tests designed to find race conditions
static void convenience_log_progress(int cycle, int totalCycles) {
  if (cycle==0) {
    std::cout << "Percent complete: ";												// begin the log line
  }
  int mod = (totalCycles/10);														// log every 10% percent complete - using mod % to output at regular intervals
  if((cycle % (mod == 0 ? 1 : mod) == 0) || (cycle == totalCycles-1)) {				// sometimes quick testing so make sure mod is not 0
    std::cout << (int)( 100.0f * (float) cycle / (float) (totalCycles-1) ) << "% ";
  }
}

class CatchMemoryLeak // This class is a utility class which tracks constructor/destructor calls (for this test) or counts times a dealloc or deallocHandle was implemented (for later tests)
{
public:
  CatchMemoryLeak() { ++s_countAllocated; }
  ~CatchMemoryLeak() { --s_countAllocated; }
  static std::atomic<int> s_countAllocated;
  static std::atomic<int> s_countDeallocs;
};
std::atomic<int> CatchMemoryLeak::s_countAllocated(0);	// counts constructor calls (+1) and destructor calls (-1) which may include double delete events
std::atomic<int> CatchMemoryLeak::s_countDeallocs(0);	// counts dealloc or dellocHandle calls - used for test 4 and test 5

} // end namespace

#endif // end #ifdef HAVE_TEUCHOSCORE_CXX11

#endif // end #ifdef TEUCHOS_GENERAL_MT_UNITTESTS_HPP
