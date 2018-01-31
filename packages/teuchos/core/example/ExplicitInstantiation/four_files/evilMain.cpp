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


#include <ostream>
#include <string>

#include "AEvil_def.hpp"
#include "BEvil_def.hpp"
#include "EvilBase_decl.hpp"
#include "EvilBase_def.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_RCPDecl.hpp"

namespace EvilPack {
template <class T> class AEvil;
template <class T> class BEvil;
}  // namespace EvilPack

template<class T>
void testEvil(const T& obj)
{

  using Teuchos::RCP;
  using Teuchos::rcp_dynamic_cast;
  using EvilPack::EvilBase;
  using EvilPack::AEvil;
  using EvilPack::BEvil;

  RCP<AEvil<T> > aEvil =
    rcp_dynamic_cast<AEvil<T> >(EvilBase<T>::createEvil("AEvil"));
  RCP<BEvil<T> > bEvil =
    rcp_dynamic_cast<BEvil<T> >(EvilBase<T>::createEvil("BEvil"));

  aEvil->soundOff(obj);
  bEvil->soundOff(obj);
  aEvil->callBEvil(*bEvil, obj);
  bEvil->callAEvil(*aEvil, obj);

}


int main()
{
  testEvil<double>(1.0);
  testEvil<int>(2);
  return 0;
}
