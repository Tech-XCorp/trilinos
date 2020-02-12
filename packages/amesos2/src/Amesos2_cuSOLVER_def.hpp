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
  auto status = CUSOLVER::cusolverSpCreate(&data_.handle);
  TEUCHOS_TEST_FOR_EXCEPTION( status != CUSOLVER::CUSOLVER_STATUS_SUCCESS,
    std::runtime_error, "cusolverSpCreate failed");

  status = CUSOLVER::cusolverSpCreateCsrcholInfo(&data_.chol_info);
  TEUCHOS_TEST_FOR_EXCEPTION( status != CUSOLVER::CUSOLVER_STATUS_SUCCESS,
    std::runtime_error, "cusolverSpCreateCsrcholInfo failed");

  auto sparse_status = CUSOLVER::cusparseCreateMatDescr(&data_.desc);
  TEUCHOS_TEST_FOR_EXCEPTION( sparse_status != CUSOLVER::CUSPARSE_STATUS_SUCCESS,
    std::runtime_error, "cusparseCreateMatDescr failed");
}


template <class Matrix, class Vector>
cuSOLVER<Matrix,Vector>::~cuSOLVER( )
{
  cusparseDestroyMatDescr(data_.desc);
  cusolverSpDestroyCsrcholInfo(data_.chol_info);
  cusolverSpDestroy(data_.handle);
}

template<class Matrix, class Vector>
int
cuSOLVER<Matrix,Vector>::preOrdering_impl()
{
#ifdef HAVE_AMESOS2_TIMERS
    Teuchos::TimeMonitor preOrderTimer(this->timers_.preOrderTime_);
#endif
  if(do_optimization()) {
    // reorder to optimize cuda
    if(data_.bReorder) {
    #ifndef HAVE_AMESOS2_METIS
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
        "Cannot reorder for cuSolver because no METIS is available.");
    #else
      const ordinal_type size = this->globalNumRows_;

      // MDM-TODO maybe set this up on device, strip diagonals on device,
      // then copy to host snd run METIS resort. Not sure if that can
      // improve efficiency here.

      // begin on host where we'll run metis reorder
      host_size_type_array host_row_ptr;
      host_ordinal_type_array host_cols;
      
      this->matrixA_->returnRowPtr_kokkos_view(host_row_ptr);
      this->matrixA_->returnColInd_kokkos_view(host_cols);

      // strip out the diagonals - metis will just crash with them in
      host_size_type_array host_strip_diag_row_ptr(
        Kokkos::ViewAllocateWithoutInitializing("host_strip_diag_row_ptr"), host_row_ptr.size());
      host_ordinal_type_array host_strip_diag_cols( // will strip diagonals so n drops by size
        Kokkos::ViewAllocateWithoutInitializing("host_strip_diag_cols"), host_cols.size() - size);
      size_type new_nnz = 0;
      for(ordinal_type i = 0; i < size; ++i) {
        host_strip_diag_row_ptr(i) = new_nnz;
        for(size_type j = host_row_ptr(i); j < host_row_ptr(i+1); ++j) {
          if (i != host_cols(j)) {
            host_strip_diag_cols(new_nnz++) = host_cols(j);
          }
        }
      }
      host_strip_diag_row_ptr(size) = new_nnz;

      // now convert to metis index type if necessary
      typedef Kokkos::View<idx_t*, HostDeviceSpaceType>          host_metis_array;
      host_metis_array host_metis_row_ptr;
      host_metis_array host_metis_cols;
      deep_copy_or_assign_view(host_metis_row_ptr, host_strip_diag_row_ptr);
      deep_copy_or_assign_view(host_metis_cols, host_strip_diag_cols);

      idx_t metis_size = size;

      // we'll get original permutations on host
      typedef Kokkos::View<idx_t*, HostDeviceSpaceType>     host_metis_array;
      host_metis_array host_perm(
        Kokkos::ViewAllocateWithoutInitializing("host_perm"), size);
      host_metis_array host_peri(
        Kokkos::ViewAllocateWithoutInitializing("host_peri"), size);

      // MDM - note cuSolver has cusolverSpXcsrmetisnd() which wraps METIS_NodeND
      // For cuSolver this could be a more elegant way to set things up.
      int err = METIS_NodeND(&metis_size, host_metis_row_ptr.data(), host_metis_cols.data(),
        NULL, NULL, host_perm.data(), host_peri.data());
      
      TEUCHOS_TEST_FOR_EXCEPTION(err != METIS_OK, std::runtime_error,
        "METIS_NodeND failed to sort matrix.");

      // put the permutations on our saved device ptrs
      // these will be used to permute x and b when we solve
      deep_copy_or_assign_view(device_perm, host_perm);
      deep_copy_or_assign_view(device_peri, host_peri);
      
      // now get matrix values on device
      device_size_type_array row_ptr;
      device_ordinal_type_array cols;
      device_value_type_array values;
      this->matrixA_->returnRowPtr_kokkos_view(row_ptr);
      this->matrixA_->returnColInd_kokkos_view(cols);
      this->matrixA_->returnValues_kokkos_view(values);
    
      // we'll permute matrix on device to a new set of arrays
      device_row_ptr_view_ = device_size_type_array(
        Kokkos::ViewAllocateWithoutInitializing("device_row_ptr_view_"), row_ptr.size());
      device_cols_view_ = device_ordinal_type_array(
        Kokkos::ViewAllocateWithoutInitializing("device_cols_view_"), cols.size());
      device_nzvals_view_ = device_value_type_array(
        Kokkos::ViewAllocateWithoutInitializing("new_values"), values.size());

      // need local access
      auto new_row_ptr = device_row_ptr_view_;
      auto new_cols = device_cols_view_;
      auto new_values = device_nzvals_view_;
      auto l_perm = device_perm;
      auto l_peri = device_peri;

      // permute row indices
      Kokkos::RangePolicy<DeviceSpaceType::execution_space> policy_row(0, row_ptr.size());
      Kokkos::parallel_scan(policy_row, KOKKOS_LAMBDA(
        ordinal_type i, size_type & update, const bool &final) {
        if(final) {
          new_row_ptr(i) = update;
        }
        if(i < size) {
          ordinal_type count = 0;
          const ordinal_type row = l_perm(i);
          for(ordinal_type k = row_ptr(row); k < row_ptr(row + 1); ++k) {
            const ordinal_type j = l_peri(cols(k)); /// col in A
            count += (i >= j); /// lower triangular
          }
          update += count;
        }
      });
      
      // permute col indices
      Kokkos::RangePolicy<DeviceSpaceType::execution_space> policy_col(0, size);
      Kokkos::parallel_for(policy_col, KOKKOS_LAMBDA(ordinal_type i) {
        const ordinal_type kbeg = new_row_ptr(i);
        const ordinal_type row = l_perm(i);
        const ordinal_type col_beg = row_ptr(row);
        const ordinal_type col_end = row_ptr(row + 1);
        const ordinal_type nk = col_end - col_beg;

        for(ordinal_type k = 0, t = 0; k < nk; ++k) {
          const ordinal_type tk = kbeg + t;
          const ordinal_type sk = col_beg + k;
          const ordinal_type j = l_peri(cols(sk));
          if(i >= j) {
            new_cols(tk) = j;
            new_values(tk) = values(sk);
            ++t;
          }
        }
      });
    #endif
    }
    else {
      this->matrixA_->returnRowPtr_kokkos_view(device_row_ptr_view_);
      this->matrixA_->returnColInd_kokkos_view(device_cols_view_);
      this->matrixA_->returnValues_kokkos_view(device_nzvals_view_);
    }
  }

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
    const int size = this->globalNumRows_;
    const int nnz = this->globalNumNonZeros_;
    const int * colIdx = device_cols_view_.data();
    const int * rowPtr = device_row_ptr_view_.data();
    auto status = CUSOLVER::cusolverSpXcsrcholAnalysis(data_.handle, size, nnz, data_.desc, rowPtr, colIdx, data_.chol_info);
    TEUCHOS_TEST_FOR_EXCEPTION( status != CUSOLVER::CUSOLVER_STATUS_SUCCESS,
      std::runtime_error, "cusolverSpXcsrcholAnalysis failed");
  }
  return status;
}


template <class Matrix, class Vector>
int
cuSOLVER<Matrix,Vector>::numericFactorization_impl()
{
  int status = 0;
  if ( this->root_ ) {
    const int size = this->globalNumRows_;
    const int nnz = this->globalNumNonZeros_;
    const cusolver_type * values = device_nzvals_view_.data();
    const int * colIdx = device_cols_view_.data();
    const int * rowPtr = device_row_ptr_view_.data();

    size_t internalDataInBytes, workspaceInBytes;
    auto status = CUSOLVER::cusolverSpDcsrcholBufferInfo(data_.handle,
                                              size, nnz, data_.desc,
                                              values, rowPtr, colIdx,
                                              data_.chol_info,
                                              &internalDataInBytes,
                                              &workspaceInBytes);
    TEUCHOS_TEST_FOR_EXCEPTION( status != CUSOLVER::CUSOLVER_STATUS_SUCCESS,
      std::runtime_error, "cusolverSpDcsrcholBufferInfo failed");

    const size_t bufsize = workspaceInBytes / sizeof(cusolver_type);
    if(bufsize > buffer_.extent(0)) {
      buffer_ = device_value_type_array(Kokkos::ViewAllocateWithoutInitializing("cusolver buf"), bufsize);
    }

    status = cusolverSpDcsrcholFactor(data_.handle,
                                         size, nnz, data_.desc,
                                         values, rowPtr, colIdx,
                                         data_.chol_info,
                                         buffer_.data());
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
  const ordinal_type nrhs = X->getGlobalNumVectors();

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

    // Some temporary clocks to track solve times
    auto startTime = std::chrono::system_clock::now();

    // permute b
    if(data_.bReorder) {
      #ifdef HAVE_AMESOS2_METIS
        // permute b
        if(this->bValues_sort_.extent(0) != this->bValues_.extent(0) || this->bValues_sort_.extent(1) != this->bValues_.extent(1)) {
          this->bValues_sort_ = device_solve_array_t(
            Kokkos::ViewAllocateWithoutInitializing("bValues_sort_"), this->bValues_.extent(0), this->bValues_.extent(1));
        }
        Kokkos::RangePolicy<DeviceSpaceType::execution_space> policy(0, this->bValues_.extent(0));
        auto l_orig_b = this->bValues_;
        auto l_sort_b = this->bValues_sort_;
        auto l_perm = this->device_perm;
        Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const ordinal_type &i) {
          for(ordinal_type j = 0; j < nrhs; ++j) {
            l_sort_b(i, j) = l_orig_b(l_perm(i), j);
          }
        });
        // be careful don't let these end up pointing to the same thing
        Kokkos::deep_copy(this->bValues_, this->bValues_sort_);
      #endif
    }

    for(ordinal_type n = 0; n < nrhs; ++n) {
      const cusolver_type * b = this->bValues_.data() + n * size;
      cusolver_type * x = this->xValues_.data() + n * size;
      function_map::solve(data_.handle, size, b, x, data_.chol_info, buffer_.data());
    }

    if(data_.bReorder) {
      #ifdef HAVE_AMESOS2_METIS
        // permute x
        if(this->xValues_sort_.extent(0) != this->xValues_.extent(0) || this->xValues_sort_.extent(1) != this->xValues_.extent(1)) {
          this->xValues_sort_ = device_solve_array_t(
            Kokkos::ViewAllocateWithoutInitializing("xValues_sort_"), this->xValues_.extent(0), this->xValues_.extent(1));;
        }
        Kokkos::RangePolicy<DeviceSpaceType::execution_space> policy(0, this->xValues_.extent(0));
        auto l_orig_x = this->xValues_;
        auto l_sort_x = this->xValues_sort_;
        auto l_peri = this->device_peri;
        Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const ordinal_type &i) {
          for(ordinal_type j = 0; j < nrhs; ++j) {
            l_sort_x(i, j) = l_orig_x(l_peri(i), j);
          }
        });
        // be careful don't let these end up pointing to the same thing
        Kokkos::deep_copy(this->xValues_, this->xValues_sort_);
      #endif
    }

    // Some temporary clocks to track solve times
    auto endTime = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    printf("Ran solve %d rhs vectors with time %d ms\n", (int) nrhs, (int) ms);
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

  if( parameterList->isParameter("Reorder") ){
    RCP<const ParameterEntryValidator> reorder_validator = valid_params->getEntry("Reorder").validator();
    parameterList->getEntry("Reorder").setValidator(reorder_validator);
  }

  data_.bReorder = parameterList->get<bool>("Reorder", true);
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

    pl->set("Reorder", true, "Whether GIDs contiguous");

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

    TEUCHOS_TEST_FOR_EXCEPTION( true, std::runtime_error,
      "cuSolver matrix reordered and the distributed matrix not resolved yet.");

/*
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
*/
  }

  return true;
}


template<class Matrix, class Vector>
const char* cuSOLVER<Matrix,Vector>::name = "cuSOLVER";


} // end namespace Amesos2

#endif  // AMESOS2_CUSOLVER_DEF_HPP
