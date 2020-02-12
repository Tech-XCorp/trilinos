// @HEADER
//
// ***********************************************************************
//
//           Amesos2: Templated Direct Sparse Solver Package
//                  Copyright 2011 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
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
//
// @HEADER


/**
   \file   Amesos2_cuSOLVER_FunctionMap.hpp
   \author Kevin Deweese <kdewees@sandia.gov>
   \date   Tue Aug 6 12:53:10 MDT 2013

   \brief  Template for providing a mechanism to map function calls to the
           correct Solver function based on the scalar type of Matrices and
           MultiVectors
*/

#ifndef AMESOS2_CUSOLVER_FUNCTIONMAP_HPP
#define AMESOS2_CUSOLVER_FUNCTIONMAP_HPP

#ifdef HAVE_TEUCHOS_COMPLEX
#include <complex>
#endif

#include "Amesos2_FunctionMap.hpp"
#include "Amesos2_cuSOLVER_TypeMap.hpp"

namespace Amesos2 {

  namespace CUSOLVER {

#    include <cuda.h>
#    include <cusolverSp.h>
#    include <cusolverDn.h>
#    include <cusparse.h>
#    include <cuComplex.h>
#    include <cusolverSp_LOWLEVEL_PREVIEW.h>  // MDM-TODO may remove - for csrcholInfo_t
  }

  template <>
  struct FunctionMap<cuSOLVER,double>
  {
    static void solve(
                 CUSOLVER::cusolverSpHandle_t handle,
                 int size,
                 const double * b,
                 double * x,
                 CUSOLVER::csrcholInfo_t & chol_info,
                 void * buffer)
    {
      auto status = CUSOLVER::cusolverSpDcsrcholSolve(
        handle, size, b, x, chol_info, buffer);

      TEUCHOS_TEST_FOR_EXCEPTION( status != CUSOLVER::CUSOLVER_STATUS_SUCCESS,
        std::runtime_error, "cusolverSpDcsrcholSolve failed with error: " << status);
    }
  };

  template <>
  struct FunctionMap<cuSOLVER,float>
  {
    static void solve(
                 CUSOLVER::cusolverSpHandle_t handle,
                 int size,
                 const float * b,
                 float * x,
                 CUSOLVER::csrcholInfo_t & chol_info,
                 void * buffer)
    {
      auto status = CUSOLVER::cusolverSpScsrcholSolve(
        handle, size, b, x, chol_info, buffer);

      TEUCHOS_TEST_FOR_EXCEPTION( status != CUSOLVER::CUSOLVER_STATUS_SUCCESS,
        std::runtime_error, "cusolverSpScsrcholSolve failed with error: " << status);
    }
  };

#ifdef HAVE_TEUCHOS_COMPLEX
  template <>
  struct FunctionMap<cuSOLVER,Kokkos::complex<double>>
  {
    static void solve(
                 CUSOLVER::cusolverSpHandle_t handle,
                 int size,
                 const cuDoubleComplex * b,
                 cuDoubleComplex * x,
                 CUSOLVER::csrcholInfo_t & chol_info,
                 void * buffer)
    {
      auto status = CUSOLVER::cusolverSpZcsrcholSolve(
        handle, size, b, x, chol_info, buffer);

      TEUCHOS_TEST_FOR_EXCEPTION( status != CUSOLVER::CUSOLVER_STATUS_SUCCESS,
        std::runtime_error, "cusolverSpZcsrcholSolve failed with error: " << status);
    }
  };
#endif

} // end namespace Amesos2

#endif  // AMESOS2_CUSOLVER_FUNCTIONMAP_HPP
