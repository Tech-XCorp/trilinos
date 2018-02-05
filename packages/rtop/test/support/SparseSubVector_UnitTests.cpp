/*
// @HEADER
// ***********************************************************************
// 
// RTOp: Interfaces and Support Software for Vector Reduction Transformation
//       Operations
//                Copyright (2006) Sandia Corporation
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
// Questions? Contact Roscoe A. Bartlett (rabartl@sandia.gov) 
// 
// ***********************************************************************
// @HEADER
*/

#include <iosfwd>

#include "RTOpPack_SparseSubVectorT.hpp"
#include "RTOpPack_Types.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_ArrayRCP.hpp"
#include "Teuchos_ArrayView.hpp"
#include "Teuchos_LocalTestingHelpers.hpp"
#include "Teuchos_TestingHelpers.hpp"
#include "Teuchos_Tuple.hpp"
#include "Teuchos_UnitTestHelpers.hpp"
#include "Teuchos_as.hpp"
#include "Teuchos_toString.hpp"


namespace RTOpPack {

//
// Test default constructor
//
TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( SparseSubVector,
  stridedConstruct, Scalar )
{
  using Teuchos::tuple;
  using Teuchos::arcpFromArray;
  using Teuchos::as;
  using Teuchos::Array;

  Array<Scalar> values = tuple<Scalar>(1.1, -1.0, 2.2, -2.0)().getConst();
  Array<Ordinal> indices = tuple<Ordinal>(1, -1, -2, 3, -3, -4)().getConst();

  SparseSubVectorT<Scalar> ssv(
    1,  // glboalOfset_in
    6,  // subDim_in
    2, // subNz_in
    arcpFromArray(values), // values_in
    2, // valuesStride_in
    arcpFromArray(indices), // indices_in
    3, // indiciesStride_in
    7, // localOffset_in
    true // isSorted_in
    );
  TEST_EQUALITY_CONST(ssv.globalOffset(), as<Ordinal>(1));
  TEST_EQUALITY_CONST(ssv.subDim(), as<Ordinal>(6));
  TEST_EQUALITY_CONST(ssv.subNz(), as<Ordinal>(2));
  TEST_COMPARE_ARRAYS(ssv.values(), values);
  TEST_EQUALITY_CONST(ssv.valuesStride(), as<Ordinal>(2));
  TEST_COMPARE_ARRAYS(ssv.indices(), indices);
  TEST_EQUALITY_CONST(ssv.indicesStride(), as<Ordinal>(3));
  TEST_EQUALITY_CONST(ssv.localOffset(), as<Ordinal>(7));
  TEST_EQUALITY_CONST(ssv.isSorted(), true);
  
  // ToDo: Add more checks ...
}
TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT_SCALAR_TYPES( SparseSubVector,
  stridedConstruct )

} // namespace RTOpPack
