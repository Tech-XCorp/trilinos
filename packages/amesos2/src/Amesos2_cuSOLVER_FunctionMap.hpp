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
  }

  template <>
  struct FunctionMap<cuSOLVER,double>
  {

    static void cusolver_solve(
                 int size,
                 int nnz,
                 const double *values,
                 const int *rowPtr,
                 const int *colIdx,
                 const double *b,
                 double tol,
                 int reorder,
                 double *x,
                 int *singularity)
{
      // MDM-TODO move this to data_ or into a new method - don't duplicate for float
      CUSOLVER::cusolverStatus_t cso;
      CUSOLVER::cusolverSpHandle_t solver_handle;
      cso = CUSOLVER::cusolverSpCreate(&solver_handle);
      TEUCHOS_TEST_FOR_EXCEPTION(cso != CUSOLVER::CUSOLVER_STATUS_SUCCESS, std::runtime_error,
        "cuSolver failed to create solver handle.");

      CUSOLVER::cusparseStatus_t csp;
      CUSOLVER::cusparseMatDescr_t descr = 0;

      CUSOLVER::cusparseCreateMatDescr(&descr);
      CUSOLVER::cusparseSetMatType(descr,CUSOLVER::CUSPARSE_MATRIX_TYPE_GENERAL);

      csp = CUSOLVER::cusparseSetMatIndexBase(descr,CUSOLVER::CUSPARSE_INDEX_BASE_ZERO);
      TEUCHOS_TEST_FOR_EXCEPTION(csp != CUSOLVER::CUSOLVER_STATUS_SUCCESS, std::runtime_error,
        "cuSolver failed to set description.");

#ifdef AMESOS2_CUSOLVER_HOST
      // Time for solve using bcsstk18.mtx (on a GTX-960)
      // cusolverSpDcsrlsvluHost     ~11s
      // cusolverSpDcsrlsvqrHost     ~11s 
      // cusolverSpDcsrlsvcholHost   ~2.4s
      cso = CUSOLVER::cusolverSpDcsrlsvcholHost(solver_handle, size, nnz, descr, values, rowPtr, colIdx, b, tol, reorder, x, singularity);
#else
      // Time for solve using bcsstk18.mtx (on a GTX-960)
      // cusolverSpDcsrlsvlu         Host only
      // cusolverSpDcsrlsvqr         ~17s 
      // cusolverSpDcsrlsvchol       ~2s
      cso = CUSOLVER::cusolverSpDcsrlsvchol(solver_handle, size, nnz, descr, values, rowPtr, colIdx, b, tol, reorder, x, singularity);
#endif

      TEUCHOS_TEST_FOR_EXCEPTION(cso != CUSOLVER::CUSOLVER_STATUS_SUCCESS, std::runtime_error,
        "cuSOLVER internal error. cusolverSpDcsrlsvqr failed.");
    }
  };

  template <>
  struct FunctionMap<cuSOLVER,float>
  {
  };

#ifdef HAVE_TEUCHOS_COMPLEX
  template <>
  struct FunctionMap<cuSOLVER,CUSOLVER::complex>
  {
  };
#endif

} // end namespace Amesos2

#endif  // AMESOS2_CUSOLVER_FUNCTIONMAP_HPP
