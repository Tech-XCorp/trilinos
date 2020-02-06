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
   \file   Amesos2_cuSolver_def.hpp
   \author John Doe <jd@sandia.gov>
   \date   Wed Jul  24 15::48:51 2013

   \brief  Definitions for the Amesos2 cuSOLVER solver interface
*/


#ifndef AMESOS2_CUSOLVER_DEF_HPP
#define AMESOS2_CUSOLVER_DEF_HPP

#include <Teuchos_Tuple.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>

#include "Amesos2_SolverCore_def.hpp"
#include "Amesos2_cuSOLVER_decl.hpp"


namespace Amesos2 {


template <class Matrix, class Vector>
cuSOLVER<Matrix,Vector>::cuSOLVER(
  Teuchos::RCP<const Matrix> A,
  Teuchos::RCP<Vector>       X,
  Teuchos::RCP<const Vector> B )
  : SolverCore<Amesos2::cuSOLVER,Matrix,Vector>(A, X, B)
{
  // MDM-cuSolver-TODO
}


template <class Matrix, class Vector>
cuSOLVER<Matrix,Vector>::~cuSOLVER( )
{
  // MDM-cuSolver-TODO
}

template<class Matrix, class Vector>
int
cuSOLVER<Matrix,Vector>::preOrdering_impl()
{
 #ifdef HAVE_AMESOS2_TIMERS
    Teuchos::TimeMonitor preOrderTimer(this->timers_.preOrderTime_);
#endif
  return(0);
}


template <class Matrix, class Vector>
int
cuSOLVER<Matrix,Vector>::symbolicFactorization_impl()
{
#ifdef HAVE_AMESOS2_TIMERS
      Teuchos::TimeMonitor symFactTimer(this->timers_.symFactTime_);
#endif

  int status = 0;
  if ( this->root_ ) {
    if(do_optimization()) {
      this->matrixA_->returnRowPtr_kokkos_view(device_row_ptr_view_);
      this->matrixA_->returnColInd_kokkos_view(device_cols_view_);
    }

    // MDM-TODO Add cuSolver symbolic
  }
  return status;

  return(0);
}


template <class Matrix, class Vector>
int
cuSOLVER<Matrix,Vector>::numericFactorization_impl()
{
  int status = 0;
  if ( this->root_ ) {
    if(do_optimization()) {
      this->matrixA_->returnValues_kokkos_view(device_nzvals_view_);
    }
    // MDM-TODO numeric
  }
  return status;
}


template <class Matrix, class Vector>
int
cuSOLVER<Matrix,Vector>::solve_impl(const Teuchos::Ptr<MultiVecAdapter<Vector> >       X,
                                   const Teuchos::Ptr<const MultiVecAdapter<Vector> > B) const
{
  using Teuchos::as;

  const global_size_type ld_rhs = this->root_ ? X->getGlobalLength() : 0;
  const size_t nrhs = X->getGlobalNumVectors();

  // don't allocate b since it's handled by the copy manager and might just be
  // be assigned, not copied anyways.
  // also don't allocate x since we will also use do_get to allocate this if
  // necessary. When a copy is not necessary we'll solve directly to the x
  // values in the MV.

  // MDM-DISCUSS If copying is necessary this is going to copy x values which we
  // don't care about. So need to refine this. Also the do_put below should
  // not be called at all for the case where the x get was just assigned.
  // I didn't fix this yet because the logic which determines if a copy
  // is necessary currently happens at a low level in the copy manager. Need
  // some form of API call at this level to determine what the outcome will be.
  {                             // Get values from RHS B
#ifdef HAVE_AMESOS2_TIMERS
    Teuchos::TimeMonitor mvConvTimer(this->timers_.vecConvTime_);
    Teuchos::TimeMonitor redistTimer(this->timers_.vecRedistTime_);
#endif

    Util::get_1d_copy_helper_kokkos_view<MultiVecAdapter<Vector>,
                             device_solve_array_t>::do_get(B, this->bValues_,
                                               as<size_t>(ld_rhs),
                                               ROOTED, this->rowIndexBase_);
    Util::get_1d_copy_helper_kokkos_view<MultiVecAdapter<Vector>,
                             device_solve_array_t>::do_get(X, this->xValues_,
                                               as<size_t>(ld_rhs),
                                               ROOTED, this->rowIndexBase_);
  }

  int ierr = 0; // returned error code

  if ( this->root_ ) {  // Do solve!
#ifdef HAVE_AMESOS2_TIMER
    Teuchos::TimeMonitor solveTimer(this->timers_.solveTime_);
#endif

    const int size = this->globalNumRows_;
    const int nnz = this->globalNumNonZeros_;
    int sing = 0;

    const cusolver_type * values = device_nzvals_view_.data();
    const int * colIdx = device_cols_view_.data();
    const int * rowPtr = device_row_ptr_view_.data();

#define USE_TEST
#ifdef USE_TEST
    CUSOLVER::cusolverSpHandle_t _handle;
    CUSOLVER::csrcholInfo_t _chol_info;
    CUSOLVER::cusparseMatDescr_t _desc;
    CUSOLVER::cusolverStatus_t status;
    CUSOLVER::cusparseStatus_t sparse_status;

    status = CUSOLVER::cusolverSpCreate(&_handle);
    TEUCHOS_TEST_FOR_EXCEPTION( status != CUSOLVER::CUSOLVER_STATUS_SUCCESS,
      std::runtime_error, "cusolverSpCreate failed");

    status = CUSOLVER::cusolverSpCreateCsrcholInfo(&_chol_info);
    TEUCHOS_TEST_FOR_EXCEPTION( status != CUSOLVER::CUSOLVER_STATUS_SUCCESS,
      std::runtime_error, "cusolverSpCreateCsrcholInfo failed");

    sparse_status = CUSOLVER::cusparseCreateMatDescr(&_desc);
    TEUCHOS_TEST_FOR_EXCEPTION( sparse_status != CUSOLVER::CUSPARSE_STATUS_SUCCESS,
      std::runtime_error, "cusparseCreateMatDescr failed");

    // symbolic
    status = CUSOLVER::cusolverSpXcsrcholAnalysis(_handle, size, nnz, _desc, rowPtr, colIdx, _chol_info);
    TEUCHOS_TEST_FOR_EXCEPTION( status != CUSOLVER::CUSOLVER_STATUS_SUCCESS,
      std::runtime_error, "cusolverSpXcsrcholAnalysis failed");

    // numeric
    size_t internalDataInBytes, workspaceInBytes;
    status = CUSOLVER::cusolverSpDcsrcholBufferInfo(_handle, 
                                              size, nnz, _desc,
                                              values, rowPtr, colIdx,
                                              _chol_info,
                                              &internalDataInBytes,
                                              &workspaceInBytes);
    TEUCHOS_TEST_FOR_EXCEPTION( status != CUSOLVER::CUSOLVER_STATUS_SUCCESS,
      std::runtime_error, "cusolverSpDcsrcholBufferInfo failed");

    const size_t bufsize = workspaceInBytes / sizeof(cusolver_type);
    if(bufsize > buffer_.extent(0)) {
      buffer_ = device_value_type_array(Kokkos::ViewAllocateWithoutInitializing("cusolver buf"), bufsize);
    }

    status = cusolverSpDcsrcholFactor(_handle,
                                         size, nnz, _desc,
                                         values, rowPtr, colIdx,
                                         _chol_info,
                                         buffer_.data());

    for(size_t n = 0; n < nrhs; ++n) {
      const cusolver_type * b = this->bValues_.data() + n * size;
      cusolver_type * x = this->xValues_.data() + n * size;

      status = CUSOLVER::cusolverSpDcsrcholSolve(
        _handle, size, b, x, _chol_info, buffer_.data());

      TEUCHOS_TEST_FOR_EXCEPTION( status != CUSOLVER::CUSOLVER_STATUS_SUCCESS,
        std::runtime_error, "cusolverSpDcsrcholSolve failed with error: " << status);
    }

    sparse_status = cusparseDestroyMatDescr(_desc);
    TEUCHOS_TEST_FOR_EXCEPTION( sparse_status != CUSOLVER::CUSPARSE_STATUS_SUCCESS,
      std::runtime_error, "cusparseDestroyMatDescr failed");

    status = cusolverSpDestroyCsrcholInfo(_chol_info);
    TEUCHOS_TEST_FOR_EXCEPTION( status != CUSOLVER::CUSOLVER_STATUS_SUCCESS,
      std::runtime_error, "cusolverSpDestroyCsrcholInfo failed");

    status = cusolverSpDestroy(_handle);
    TEUCHOS_TEST_FOR_EXCEPTION( status != CUSOLVER::CUSOLVER_STATUS_SUCCESS,
      std::runtime_error, "cusolverSpDestroy failed");

#else
    // for now just get a solution which works for multiple vectors
    // then later we need to see how to batch solver
    for(size_t n = 0; n < nrhs; ++n) {
      const cusolver_type * b = &this->bValues_.data()[n*size];
      cusolver_type * x = &this->xValues_.data()[n*size];
      function_map::cusolver_solve(size, nnz, values, rowPtr, colIdx, b, 0.0, 0, x, &sing);
    }
#endif
  }

  /* All processes should have the same error code */
  Teuchos::broadcast(*(this->getComm()), 0, &ierr);

  TEUCHOS_TEST_FOR_EXCEPTION( ierr != 0, std::runtime_error,
    "cusolver has error code: " << ierr );

  /* Update X's global values */
  {
#ifdef HAVE_AMESOS2_TIMERS
    Teuchos::TimeMonitor redistTimer(this->timers_.vecRedistTime_);
#endif

    // This will do nothing is if the target view matches the src view, which
    // can be the case if the memory spaces match. See comments above for do_get.
    Util::template put_1d_data_helper_kokkos_view<
      MultiVecAdapter<Vector>,device_solve_array_t>::do_put(X, xValues_,
                                        as<size_t>(ld_rhs),
                                        ROOTED, this->rowIndexBase_);
  }

  return(ierr);
}


template <class Matrix, class Vector>
bool
cuSOLVER<Matrix,Vector>::matrixShapeOK_impl() const
{
  return( this->matrixA_->getGlobalNumRows() == this->matrixA_->getGlobalNumCols() );
}


template <class Matrix, class Vector>
void
cuSOLVER<Matrix,Vector>::setParameters_impl(const Teuchos::RCP<Teuchos::ParameterList> & parameterList )
{
  using Teuchos::RCP;
  using Teuchos::getIntegralValue;
  using Teuchos::ParameterEntryValidator;

  RCP<const Teuchos::ParameterList> valid_params = getValidParameters_impl();


  if( parameterList->isParameter("Trans") ){
    RCP<const ParameterEntryValidator> trans_validator = valid_params->getEntry("Trans").validator();
    parameterList->getEntry("Trans").setValidator(trans_validator);

  }
}


template <class Matrix, class Vector>
Teuchos::RCP<const Teuchos::ParameterList>
cuSOLVER<Matrix,Vector>::getValidParameters_impl() const
{
  using std::string;
  using Teuchos::tuple;
  using Teuchos::ParameterList;
  using Teuchos::EnhancedNumberValidator;
  using Teuchos::setStringToIntegralParameter;
  using Teuchos::stringToIntegralParameterEntryValidator;

  static Teuchos::RCP<const Teuchos::ParameterList> valid_params;

  if( is_null(valid_params) ){
    Teuchos::RCP<Teuchos::ParameterList> pl = Teuchos::parameterList();
    valid_params = pl;
  }

  return valid_params;
}


template <class Matrix, class Vector>
bool
cuSOLVER<Matrix,Vector>::do_optimization() const {
  return (this->root_ && (this->matrixA_->getComm()->getSize() == 1));
}


template <class Matrix, class Vector>
bool
cuSOLVER<Matrix,Vector>::loadA_impl(EPhase current_phase)
{
  if(current_phase == SOLVE) {
    return(false);
  }

  if(!do_optimization()) {
#ifdef HAVE_AMESOS2_TIMERS
  Teuchos::TimeMonitor convTimer(this->timers_.mtxConvTime_);
#endif

    // Note views are allocated but eventually we should remove this.
    // The internal copy manager will decide if we can assign or deep_copy
    // and then allocate if necessary. However the GPU solvers are serial right
    // now so I didn't complete refactoring the matrix code for the parallel
    // case. If we added that later, we should have it hooked up to the copy
    // manager and then these allocations can go away.
    if( this->root_ ) {
      device_nzvals_view_ = device_value_type_array(
        Kokkos::ViewAllocateWithoutInitializing("nzvals"), this->globalNumNonZeros_);
      device_cols_view_ = device_ordinal_type_array(
        Kokkos::ViewAllocateWithoutInitializing("colind"), this->globalNumNonZeros_);
      device_row_ptr_view_ = device_size_type_array(
        Kokkos::ViewAllocateWithoutInitializing("rowptr"), this->globalNumRows_ + 1);
    }

    typename device_size_type_array::value_type nnz_ret = 0;
    {
  #ifdef HAVE_AMESOS2_TIMERS
      Teuchos::TimeMonitor mtxRedistTimer( this->timers_.mtxRedistTime_ );
  #endif

      TEUCHOS_TEST_FOR_EXCEPTION( this->rowIndexBase_ != this->columnIndexBase_,
                          std::runtime_error,
                          "Row and column maps have different indexbase ");


      Util::get_crs_helper_kokkos_view<MatrixAdapter<Matrix>,
        device_value_type_array, device_ordinal_type_array, device_size_type_array>::do_get(
                                                      this->matrixA_.ptr(),
                                                      device_nzvals_view_,
                                                      device_cols_view_,
                                                      device_row_ptr_view_,
                                                      nnz_ret,
                                                      ROOTED, ARBITRARY,
                                                      this->columnIndexBase_);
    }
  }

  return true;
}


template<class Matrix, class Vector>
const char* cuSOLVER<Matrix,Vector>::name = "cuSOLVER";


} // end namespace Amesos2

#endif  // AMESOS2_CUSOLVER_DEF_HPP
