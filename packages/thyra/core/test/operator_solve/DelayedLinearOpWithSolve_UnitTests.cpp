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


#include <ostream>

#include "OperatorSolveHelpers.hpp"
#include "Teuchos_Describable.hpp"
#include "Teuchos_ENull.hpp"
#include "Teuchos_FancyOStream.hpp"
#include "Teuchos_LocalTestingHelpers.hpp"
#include "Teuchos_Ptr.hpp"
#include "Teuchos_PtrDecl.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_RCPDecl.hpp"
#include "Teuchos_RCPNode.hpp"
#include "Teuchos_TestingHelpers.hpp"
#include "Teuchos_UnitTestHelpers.hpp"
#include "Teuchos_as.hpp"
#include "Teuchos_toString.hpp"
#include "Thyra_DefaultLinearOpSource_decl.hpp"
#include "Thyra_DefaultSerialDenseLinearOpWithSolveFactory_decl.hpp"
#include "Thyra_DefaultSpmdVectorSpace_decl.hpp"
#include "Thyra_DelayedLinearOpWithSolve_decl.hpp"
#include "Thyra_LinearOpTester_decl.hpp"
#include "Thyra_LinearOpWithSolveTester_decl.hpp"
#include "Thyra_OperatorVectorTypes.hpp"
#include "Thyra_SolveSupportTypes.hpp"
#include "Thyra_UnitTestHelpers.hpp"


namespace Thyra {


//
// Helper code and declarations
//


template <class Scalar> class MultiVectorBase;
template <class Scalar> class VectorSpaceBase;

using Teuchos::as;
using Teuchos::null;
using Teuchos::RCP;
using Teuchos::inOutArg;


//
// Unit Tests
//


TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( DelayedLinearOpWithSolve, defaultConstruct,
  Scalar )
{
  const RCP<DelayedLinearOpWithSolve<Scalar> > dlows =
    delayedLinearOpWithSolve<Scalar>();
  TEST_ASSERT(nonnull(dlows));
  TEST_EQUALITY_CONST(dlows->range(), null);
  TEST_EQUALITY_CONST(dlows->domain(), null);
  out << "dlows = " << *dlows;
}
THYRA_UNIT_TEST_TEMPLATE_1_INSTANT_SCALAR_TYPES( DelayedLinearOpWithSolve,
  defaultConstruct )


TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( DelayedLinearOpWithSolve, basic,
  Scalar )
{

  // typedef Teuchos::ScalarTraits<Scalar> ST; // nused

  const Ordinal dim = 4;

  const RCP<const VectorSpaceBase<Scalar> > vs =
    defaultSpmdVectorSpace<Scalar>(dim);

  const RCP<const MultiVectorBase<Scalar> > M =
    createNonsingularMultiVector(vs);

  const RCP<DelayedLinearOpWithSolve<Scalar> > dlows =
    delayedLinearOpWithSolve<Scalar>(
      defaultLinearOpSource<Scalar>(M),
      defaultSerialDenseLinearOpWithSolveFactory<Scalar>()
      );

  TEST_ASSERT(nonnull(dlows));
  TEST_ASSERT(dlows->range()->isCompatible(*vs));
  TEST_ASSERT(dlows->domain()->isCompatible(*vs));
  out << "dlows = " << *dlows;

  Thyra::LinearOpTester<Scalar> linearOpTester;
  const bool checkOpResult = linearOpTester.check(*dlows, inOutArg(out));
  TEST_ASSERT(checkOpResult);

  Thyra::LinearOpWithSolveTester<Scalar> linearOpWithSolveTester;
  const bool checkSolveResult = linearOpWithSolveTester.check(*dlows, &out);
  TEST_ASSERT(checkSolveResult);

}
THYRA_UNIT_TEST_TEMPLATE_1_INSTANT_SCALAR_TYPES( DelayedLinearOpWithSolve,
  basic )


} // namespace Thyra
