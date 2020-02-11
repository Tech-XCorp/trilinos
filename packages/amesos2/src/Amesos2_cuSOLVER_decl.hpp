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
  \file   Amesos2_cuSOLVER_decl.hpp
  \author John Doe <jd@sandia.gov>
  \date   Tue Aug 27 17:06:53 2013

  \brief  Amesos2 cuSOLVER declarations.
*/


#ifndef AMESOS2_CUSOLVER_DECL_HPP
#define AMESOS2_CUSOLVER_DECL_HPP

// Temporary measure to switch cuSolver between host and device solve mode
// This dictates what space x and b live in 
//  #define AMESOS2_CUSOLVER_HOST

#include "Amesos2_SolverTraits.hpp"
#include "Amesos2_SolverCore.hpp"
#include "Amesos2_cuSOLVER_FunctionMap.hpp"


namespace Amesos2 {


/** \brief Amesos2 interface to cuSOLVER.
 *
 * See the \ref superlu_parameters "summary of SuperLU parameters"
 * supported by this Amesos2 interface.
 *
 * \ingroup amesos2_solver_interfaces
 */
template <class Matrix,
          class Vector>
class cuSOLVER : public SolverCore<Amesos2::cuSOLVER, Matrix, Vector>
{
  friend class SolverCore<Amesos2::cuSOLVER,Matrix,Vector>; // Give our base access
                                                          // to our private
                                                          // implementation funcs
public:

  /// Name of this solver interface.
  static const char* name;      // declaration. Initialization outside.

  typedef cuSOLVER<Matrix,Vector>                                       type;
  typedef SolverCore<Amesos2::cuSOLVER,Matrix,Vector>             super_type;

  // Since typedef's are not inheritted, go grab them
  typedef typename super_type::scalar_type                    scalar_type;
  typedef typename super_type::local_ordinal_type      local_ordinal_type;
  typedef typename super_type::global_ordinal_type    global_ordinal_type;
  typedef typename super_type::global_size_type          global_size_type;
  typedef typename super_type::node_type                        node_type;

  typedef TypeMap<Amesos2::cuSOLVER,scalar_type>                    type_map;

  typedef typename type_map::type                             cusolver_type;
  typedef typename type_map::magnitude_type                  magnitude_type;

  typedef FunctionMap<Amesos2::cuSOLVER,cusolver_type>         function_map;

#ifdef AMESOS2_CUSOLVER_HOST 
  typedef Kokkos::DefaultHostExecutionSpace                 DeviceSpaceType;
#else
  #ifdef KOKKOS_ENABLE_CUDA
    // special case - use UVM Off not UVM on to test the current targets
    typedef Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>  DeviceSpaceType;
  #else
    typedef Kokkos::DefaultExecutionSpace                    DeviceSpaceType;
  #endif
#endif

  typedef int                                                  ordinal_type;
  typedef int                                                     size_type;

  typedef Kokkos::View<size_type*, DeviceSpaceType>       device_size_type_array;
  typedef Kokkos::View<ordinal_type*, DeviceSpaceType> device_ordinal_type_array;
  typedef Kokkos::View<cusolver_type*, DeviceSpaceType>  device_value_type_array;

  typedef Kokkos::DefaultHostExecutionSpace                    HostDeviceSpaceType;
  typedef Kokkos::View<size_type*, HostDeviceSpaceType>       host_size_type_array;
  typedef Kokkos::View<ordinal_type*, HostDeviceSpaceType> host_ordinal_type_array;


#ifdef HAVE_AMESOS2_METIS // data for reordering
  typedef Kokkos::View<idx_t*, DeviceSpaceType>               device_metis_array;
  device_metis_array device_perm;
  device_metis_array device_peri;
#endif
  
  /// \name Constructor/Destructor methods
  //@{

  /**
   * \brief Initialize from Teuchos::RCP.
   *
   * \warning Should not be called directly!  Use instead
   * Amesos2::create() to initialize a cuSOLVER interface.
   */
  cuSOLVER(Teuchos::RCP<const Matrix> A,
          Teuchos::RCP<Vector>       X,
          Teuchos::RCP<const Vector> B);


  /// Destructor
  ~cuSOLVER( );

  //@}

private:

  /**
   * \brief Performs pre-ordering on the matrix to increase efficiency.
   */
  int preOrdering_impl();


  /**
   * \brief Perform symbolic factorization of the matrix using cuSOLVER.
   *
   * Called first in the sequence before numericFactorization.
   *
   * \throw std::runtime_error cuSOLVER is not able to factor the matrix.
   */
  int symbolicFactorization_impl();


  /**
   * \brief cuSOLVER specific numeric factorization
   *
   * \throw std::runtime_error cuSOLVER is not able to factor the matrix
   */
  int numericFactorization_impl();


  /**
   * \brief cuSOLVER specific solve.
   *
   * Uses the symbolic and numeric factorizations, along with the RHS
   * vector \c B to solve the sparse system of equations.  The
   * solution is placed in X.
   *
   * \throw std::runtime_error cuSOLVER is not able to solve the system.
   *
   * \callgraph
   */
  int solve_impl(const Teuchos::Ptr<MultiVecAdapter<Vector> >       X,
                 const Teuchos::Ptr<const MultiVecAdapter<Vector> > B) const;


  /**
   * \brief Determines whether the shape of the matrix is OK for this solver.
   */
  bool matrixShapeOK_impl() const;


  /**
   * Currently, the following cuSOLVER parameters/options are
   * recognized and acted upon:
   *
   * MDM-TODO Update docs
   * <ul>
   *   <li> \c "Trans" : { \c "NOTRANS" | \c "TRANS" |
   *     \c "CONJ" }.  Specifies whether to solve with the transpose system.</li>
   *   <li> \c "Equil" : { \c true | \c false }.  Specifies whether
   *     the solver to equilibrate the matrix before solving.</li>
   *   <li> \c "IterRefine" : { \c "NO" | \c "SLU_SINGLE" | \c "SLU_DOUBLE" | \c "EXTRA"
   *     }. Specifies whether to perform iterative refinement, and in
   *     what precision to compute the residual.</li>
   *   <li> \c "SymmetricMode" : { \c true | \c false }.</li>
   *   <li> \c "DiagPivotThresh" : \c double value. Specifies the threshold
   *     used for a diagonal to be considered an acceptable pivot.</li>
   *   <li> \c "ColPerm" which takes one of the following:
   *     <ul>
   *     <li> \c "NATURAL" : natural ordering.</li>
   *     <li> \c "MMD_AT_PLUS_A" : minimum degree ordering on the structure of
   *       \f$ A^T + A\f$ .</li>
   *     <li> \c "MMD_ATA" : minimum degree ordering on the structure of
   *       \f$ A T A \f$ .</li>
   *     <li> \c "COLAMD" : approximate minimum degree column ordering.
   *       (default)</li>
   *     </ul>
   * </ul>
   */
  void setParameters_impl(
    const Teuchos::RCP<Teuchos::ParameterList> & parameterList );


  /**
   * Hooked in by Amesos2::SolverCore parent class.
   *
   * \return a const Teuchos::ParameterList of all valid parameters for this
   * solver.
   */
  Teuchos::RCP<const Teuchos::ParameterList> getValidParameters_impl() const;


  /**
   * \brief Reads matrix data into internal structures
   *
   * \param [in] cur rent_phase an indication of which solution phase this
   *                           load is being performed for.
   *
   * \return \c true if the matrix was loaded, \c false if not
   */
  bool loadA_impl(EPhase current_phase);

  /**
   * \brief can we optimize size_type and ordinal_type for straight pass through
   */
  bool do_optimization() const;

  // struct holds all data necessary to make a superlu factorization or solve call
  mutable struct cuSolverData {
    CUSOLVER::cusolverSpHandle_t handle;
    CUSOLVER::csrcholInfo_t chol_info;
    CUSOLVER::cusparseMatDescr_t desc;
  } data_;

  typedef Kokkos::View<cusolver_type**, Kokkos::LayoutLeft, DeviceSpaceType> device_solve_array_t;

  mutable device_solve_array_t xValues_;
  mutable device_solve_array_t bValues_;
  mutable device_value_type_array buffer_;

  device_value_type_array device_nzvals_view_;
  device_size_type_array device_row_ptr_view_;
  device_ordinal_type_array device_cols_view_;

  bool bReorder_; // temp - probably delete this later - used to test both ways
};                              // End class cuSOLVER

template <>
struct solver_traits<cuSOLVER> {
#ifdef HAVE_TEUCHOS_COMPLEX
  typedef Meta::make_list5<float,
			   double,
                           std::complex<double>,
                           Kokkos::complex<double>,
                           CUSOLVER::complex> supported_scalars;
#else
  typedef Meta::make_list2<float, double> supported_scalars;
#endif
};

template <typename Scalar, typename LocalOrdinal, typename ExecutionSpace>
struct solver_supports_matrix<cuSOLVER,
  KokkosSparse::CrsMatrix<Scalar, LocalOrdinal, ExecutionSpace>> {
  static const bool value = true;
};

} // end namespace Amesos2

#endif  // AMESOS2_CUSOLVER_DECL_HPP
