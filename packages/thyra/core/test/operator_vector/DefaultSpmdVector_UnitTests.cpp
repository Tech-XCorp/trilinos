/*
// @HEADER
// ***********************************************************************
// 
//    Thyra: Interfaces and Support for Abstract Numerical Algorithms
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
// Questions? Contact Roscoe A. Bartlett (bartlettra@ornl.gov) 
// 
// ***********************************************************************
// @HEADER
*/


#include <stdlib.h>
#include <algorithm>
#include <sstream>

#include "Teuchos_ArrayRCP.hpp"
#include "Teuchos_ArrayRCPDecl.hpp"
#include "Teuchos_ArrayView.hpp"
#include "Teuchos_Comm.hpp"
#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_ConfigDefs.hpp"
#include "Teuchos_DefaultComm.hpp"
#include "Teuchos_ENull.hpp"
#include "Teuchos_FancyOStream.hpp"
#include "Teuchos_LocalTestingHelpers.hpp"
#include "Teuchos_Ptr.hpp"
#include "Teuchos_PtrDecl.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_RCPDecl.hpp"
#include "Teuchos_ScalarTraitsDecl.hpp"
#include "Teuchos_StaticSetupMacro.hpp"
#include "Teuchos_TestingHelpers.hpp"
#include "Teuchos_UnitTestHelpers.hpp"
#include "Teuchos_UnitTestRepository.hpp"
#include "Teuchos_as.hpp"
#include "Teuchos_toString.hpp"
#include "Thyra_DefaultSpmdVectorSpace_decl.hpp"
#include "Thyra_OperatorVectorTypes.hpp"
#include "Thyra_VectorSpaceBase_decl.hpp"
#include "Thyra_VectorStdOps_decl.hpp"


namespace Thyra {


//
// Helper code and declarations
//


template <class Scalar> class SpmdMultiVectorBase;
template <class Scalar> class VectorBase;

using Teuchos::as;
using Teuchos::null;
using Teuchos::rcp_dynamic_cast;
using Teuchos::inoutArg;
using Teuchos::outArg;


int g_localDim = 3;


TEUCHOS_STATIC_SETUP()
{
  Teuchos::UnitTestRepository::getCLP().setOption(
    "local-dim", &g_localDim, "Local dimension of each vector." );
}


template<class Scalar>
RCP<VectorSpaceBase<Scalar> > 
createSpmdVectorSpace(const Teuchos_Ordinal localDim)
{
  return defaultSpmdVectorSpace<Scalar>(
    Teuchos::DefaultComm<Teuchos_Ordinal>::getComm(),
    localDim, -1 );
}


//
// Unit Tests
//


TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( DefaultSpmdVector, getMultiVectorLocalData,
  Scalar )
{
  out << "Test that we can grab MV data from Vector ...\n";

  typedef typename ScalarTraits<Scalar>::magnitudeType ScalarMag;
  RCP<const VectorSpaceBase<Scalar> > vs = createSpmdVectorSpace<Scalar>(g_localDim);
  const int procRank = Teuchos::DefaultComm<Ordinal>::getComm()->getRank();
  RCP<VectorBase<Scalar> > v = createMember(*vs);
  const ScalarMag tol = 100.0*ScalarTraits<Scalar>::eps();
  const Ordinal globalOffset = procRank * g_localDim;

  out << "Get non-const MV local data and set it ...\n";
  {
    ArrayRCP<Scalar> localValues;
    Ordinal leadingDim = -1;
    rcp_dynamic_cast<SpmdMultiVectorBase<Scalar> >(v,true)->getNonconstLocalData(
      outArg(localValues), outArg(leadingDim));
    TEST_EQUALITY(localValues.size(), g_localDim);
    TEST_EQUALITY(leadingDim, g_localDim);
    for (int i = 0; i < localValues.size(); ++i) {
      localValues[i] = globalOffset + i + 1;
    } 
  }
  const Ordinal n = vs->dim();
  TEST_FLOATING_EQUALITY(sum<Scalar>(*v), as<Scalar>((n*(n+1))/2.0), tol);

  out << "Get const MV local data and check it ...\n";
  {
    ArrayRCP<const Scalar> localValues;
    Ordinal leadingDim = -1;
    rcp_dynamic_cast<const SpmdMultiVectorBase<Scalar> >(v,true)->getLocalData(
      outArg(localValues), outArg(leadingDim));
    TEST_EQUALITY(localValues.size(), g_localDim);
    TEST_EQUALITY(leadingDim, g_localDim);
    for (int i = 0; i < localValues.size(); ++i) {
      TEST_EQUALITY(localValues[i], as<Scalar>(globalOffset + i + 1));
    } 
  }
}
TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT_SCALAR_TYPES( DefaultSpmdVector,
  getMultiVectorLocalData )


} // namespace Thyra
