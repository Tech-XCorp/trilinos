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
// Questions? Contact Sivasankaran Rajamanickam (srajama@sandia.gov)
//
// ***********************************************************************
//
// @HEADER

#ifndef AMESOS2_TACHO_DEF_HPP
#define AMESOS2_TACHO_DEF_HPP

#include <Teuchos_Tuple.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>

#include "Amesos2_SolverCore_def.hpp"
#include "Amesos2_Tacho_decl.hpp"
#include "Amesos2_Util.hpp"

namespace Amesos2 {

template <class Matrix, class Vector>
TachoSolver<Matrix,Vector>::TachoSolver(
  Teuchos::RCP<const Matrix> A,
  Teuchos::RCP<Vector>       X,
  Teuchos::RCP<const Vector> B )
  : SolverCore<Amesos2::TachoSolver,Matrix,Vector>(A, X, B)
{

  int sym = 3;
  int posdef = 1;
  int small_problem_thres = 1024;

// temp try some values
data_.solver.setMaxNumberOfSuperblocks(32);
data_.solver.setSmallProblemThresholdsize(small_problem_thres);
data_.solver.setBlocksize(64);
data_.solver.setPanelsize(32);
data_.solver.setMatrixType(sym, posdef);

}


template <class Matrix, class Vector>
TachoSolver<Matrix,Vector>::~TachoSolver( )
{
}

template <class Matrix, class Vector>
std::string
TachoSolver<Matrix,Vector>::description() const
{
  std::ostringstream oss;
  oss << "Tacho solver interface";
  return oss.str();
}


template<class Matrix, class Vector>
int
TachoSolver<Matrix,Vector>::preOrdering_impl()
{
#ifdef HAVE_AMESOS2_METIS
  const ordinal_type size = this->globalNumRows_;

  host_size_type_array row_ptr;
  host_ordinal_type_array cols;
  host_value_type_array values;
  
  this->matrixA_->returnRowPtr_kokkos_view(row_ptr);
  this->matrixA_->returnColInd_kokkos_view(cols);
  this->matrixA_->returnValues_kokkos_view(values);

  host_metis_array metis_row_ptr;
  host_metis_array metis_cols;
//  deep_copy_or_assign_view(metis_row_ptr, row_ptr);
//  deep_copy_or_assign_view(metis_col_idx, cols);

// TEMP!!!!
  deep_copy(metis_row_ptr, row_ptr);
  deep_copy(metis_cols, cols);

  idx_t metis_size = size;

  perm = host_metis_array(
    Kokkos::ViewAllocateWithoutInitializing("Metis::PermutationArray"), size);
  peri = host_metis_array(
    Kokkos::ViewAllocateWithoutInitializing("Metis::InvPermutationArray"), size);
  
// TEMP!!!!!!
  perm = host_metis_array(
    ("Metis::PermutationArray"), size);
  peri = host_metis_array(
    ("Metis::InvPermutationArray"), size);
  
  printf("metis_row_ptr (%d): ", (int) metis_row_ptr.size());
  for(int n = 0; n < (int) metis_row_ptr.size(); ++n) {
    printf("%d ", (int) metis_row_ptr(n));
  }
  printf("\n");
  printf("metis_cols (%d): ", (int) metis_cols.size());
  for(int n = 0; n < (int) metis_cols.size(); ++n) {
    printf("%d ", (int) metis_cols(n));
  }
  printf("\n");
  
  // MDM - note cuSolver has cusolverSpXcsrmetisnd() which wraps METIS_NodeND
  // For cuSolver this could be a more elegant way to set things up.
  int err = METIS_NodeND(&metis_size, metis_row_ptr.data(), metis_cols.data(),
    NULL, NULL, perm.data(), peri.data());

  TEUCHOS_TEST_FOR_EXCEPTION(err != METIS_OK, std::runtime_error,
    "METIS_NodeND failed to sort matrix.");
      
/*
  // First we'll convert on host but we could mirror perm and peri to the exec
  // space of the matrix and then do this there
  typedef  Kokkos::DefaultHostExecutionSpace device_exec_memory_space;
  
  host_size_type_array new_row_ptr(
    Kokkos::ViewAllocateWithoutInitializing("new_row_ptr"), row_ptr.size());
  host_ordinal_type_array new_cols(
    Kokkos::ViewAllocateWithoutInitializing("new_cols"), cols.size());
  host_value_type_array new_values(
    Kokkos::ViewAllocateWithoutInitializing("new_values"), values.size());
  
  auto device_perm = Kokkos::create_mirror_view(device_exec_memory_space(),  perm);
  Kokkos::deep_copy(device_perm,  perm);
  auto device_peri = Kokkos::create_mirror_view(device_exec_memory_space(), peri);
  Kokkos::deep_copy(device_peri, peri);
  
  {  /// permute row indices (exclusive scan)
    Kokkos::RangePolicy<device_exec_memory_space> policy(0, row_ptr.size());
    Kokkos::parallel_scan(policy, KOKKOS_LAMBDA(
      ordinal_type i, size_type & update, const bool &final) {
      if (final) {
        new_row_ptr(i) = update;
      }

      if (i < size) {
        ordinal_type count = 0;
        const ordinal_type row = device_perm(i);
        for(ordinal_type k = row_ptr(row); k < row_ptr(row + 1); ++k) {
          const ordinal_type j = device_peri(cols(k)); /// col in A
          count += (i >= j); /// lower triangular
        }
        update += count;
      }
    });
    Kokkos::fence();
  }
  {  /// permute col indices (do not sort)
    Kokkos::RangePolicy<device_exec_memory_space> policy(0, size);
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(ordinal_type i) {
      const ordinal_type kbeg = new_row_ptr(i);
      const ordinal_type row = device_perm(i);
      const ordinal_type col_beg = row_ptr(row);
      const ordinal_type col_end = (row < row_ptr.size() - 1) ? row_ptr(row + 1) : cols.size();
      const ordinal_type nk = col_end - col_beg;

      for(ordinal_type k = 0, t = 0; k < nk; ++k) {
        const ordinal_type tk = kbeg + t;
        const ordinal_type sk = col_beg + k;
        const ordinal_type j = device_peri(cols(sk));
        if (i >= j) {
          new_cols(tk) = j;
          new_values(tk) = values(sk);
          ++t;
        }
      }
    });
    Kokkos::fence();
  }
*/

#ifdef HAVE_AMESOS2_METIS
/*
  // permute x
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const ordinal_type &i) {
    for(ordinal_type j = 0; j < n; ++j) {
      A(i, j) = B(p(i), j);
    }
  });
*/
#endif

#endif

  return(0);
  
}

template <class Matrix, class Vector>
int
TachoSolver<Matrix,Vector>::symbolicFactorization_impl()
{
  int status = 0;
  if ( this->root_ ) {
    if(do_optimization()) {
      this->matrixA_->returnRowPtr_kokkos_view(host_row_ptr_view_);
      this->matrixA_->returnColInd_kokkos_view(host_cols_view_);
    }

    // TODO: Confirm param options
    // data_.solver.setMaxNumberOfSuperblocks(data_.max_num_superblocks);

    // Symbolic factorization currently must be done on host
    data_.solver.analyze(this->globalNumCols_, host_row_ptr_view_, host_cols_view_);
  }
  return status;
}


template <class Matrix, class Vector>
int
TachoSolver<Matrix,Vector>::numericFactorization_impl()
{
  int status = 0;
  if ( this->root_ ) {
    if(do_optimization()) {
      this->matrixA_->returnValues_kokkos_view(device_nzvals_view_);
    }
    data_.solver.factorize(device_nzvals_view_);
  }
  return status;
}

template <class Matrix, class Vector>
int
TachoSolver<Matrix,Vector>::solve_impl(const Teuchos::Ptr<MultiVecAdapter<Vector> > X,
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
  {                             // Get values from RHS B
#ifdef HAVE_AMESOS2_TIMERS
    Teuchos::TimeMonitor mvConvTimer(this->timers_.vecConvTime_);
    Teuchos::TimeMonitor redistTimer(this->timers_.vecRedistTime_);
#endif
    Util::get_1d_copy_helper_kokkos_view<MultiVecAdapter<Vector>,
                             device_solve_array_t>::do_get(B, this->bValues_,
                                               as<size_t>(ld_rhs),
                                               ROOTED, this->rowIndexBase_);

    // If it's not a match and we copy instead of ptr assignement, we will
    // copy the x values here when we just wanted to get uninitialized space.
    // MDM-DISCUSS Need to decide how to request the underlying API to learn
    // whether we will need to copy or not.
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
    // Bump up the workspace size if needed
    if (workspace_.extent(0) < this->globalNumRows_ || workspace_.extent(1) < nrhs) {
      workspace_ = device_solve_array_t(
        Kokkos::ViewAllocateWithoutInitializing("t"), this->globalNumRows_, nrhs);
    }

    data_.solver.solve(xValues_, bValues_, workspace_);
  }

  /* All processes should have the same error code */
  Teuchos::broadcast(*(this->getComm()), 0, &ierr);

  TEUCHOS_TEST_FOR_EXCEPTION( ierr != 0, std::runtime_error,
    "tacho_solve has error code: " << ierr );

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
TachoSolver<Matrix,Vector>::matrixShapeOK_impl() const
{
  // Tacho can only apply the solve routines to square matrices
  return( this->matrixA_->getGlobalNumRows() == this->matrixA_->getGlobalNumCols() );
}


template <class Matrix, class Vector>
void
TachoSolver<Matrix,Vector>::setParameters_impl(const Teuchos::RCP<Teuchos::ParameterList> & parameterList )
{
  RCP<const Teuchos::ParameterList> valid_params = getValidParameters_impl();

  // TODO: Confirm param options
  // data_.num_kokkos_threads = parameterList->get<int>("kokkos-threads", 1);
  // data_.max_num_superblocks = parameterList->get<int>("max-num-superblocks", 4);
}


template <class Matrix, class Vector>
Teuchos::RCP<const Teuchos::ParameterList>
TachoSolver<Matrix,Vector>::getValidParameters_impl() const
{
  static Teuchos::RCP<const Teuchos::ParameterList> valid_params;

  if( is_null(valid_params) ){
    Teuchos::RCP<Teuchos::ParameterList> pl = Teuchos::parameterList();

    // TODO: Confirm param options
    // pl->set("kokkos-threads", 1, "Number of threads");
    // pl->set("max-num-superblocks", 4, "Max number of superblocks");

    valid_params = pl;
  }

  return valid_params;
}

template <class Matrix, class Vector>
bool
TachoSolver<Matrix,Vector>::do_optimization() const {
  return (this->root_ && (this->matrixA_->getComm()->getSize() == 1));
}

template <class Matrix, class Vector>
bool
TachoSolver<Matrix,Vector>::loadA_impl(EPhase current_phase)
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
    if(this->root_) {
      device_nzvals_view_ = device_value_type_array(
        Kokkos::ViewAllocateWithoutInitializing("nzvals"), this->globalNumNonZeros_);
      host_cols_view_ = host_ordinal_type_array(
        Kokkos::ViewAllocateWithoutInitializing("colind"), this->globalNumNonZeros_);
      host_row_ptr_view_ = host_size_type_array(
        Kokkos::ViewAllocateWithoutInitializing("rowptr"), this->globalNumRows_ + 1);
    }

    typename host_size_type_array::value_type nnz_ret = 0;
    {
  #ifdef HAVE_AMESOS2_TIMERS
      Teuchos::TimeMonitor mtxRedistTimer( this->timers_.mtxRedistTime_ );
  #endif

      TEUCHOS_TEST_FOR_EXCEPTION( this->rowIndexBase_ != this->columnIndexBase_,
                          std::runtime_error,
                          "Row and column maps have different indexbase ");

      Util::get_crs_helper_kokkos_view<MatrixAdapter<Matrix>,
        device_value_type_array, host_ordinal_type_array, host_size_type_array>::do_get(
                                                      this->matrixA_.ptr(),
                                                      device_nzvals_view_,
                                                      host_cols_view_,
                                                      host_row_ptr_view_,
                                                      nnz_ret,
                                                      ROOTED, ARBITRARY,
                                                      this->columnIndexBase_);
    }
  }

  return true;
}


template<class Matrix, class Vector>
const char* TachoSolver<Matrix,Vector>::name = "Tacho";


} // end namespace Amesos2

#endif  // AMESOS2_TACHO_DEF_HPP
