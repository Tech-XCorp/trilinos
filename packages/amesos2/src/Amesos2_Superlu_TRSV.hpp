/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
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
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

// Note this file is essentially a copy of KokkosSparse_sptrsv.hpp and may be
// temporary. It was added because Amesos2 currently namespaces the superlu
// includes. The only change to this file is to add the SLU:: namespace prefix.
#ifndef AMESOS2_SUPERLU_TRSV_HPP_
#define AMESOS2_SUPERLU_TRSV_HPP_

#ifdef KOKKOSKERNELS_ENABLE_TPL_SUPERLU

#include "KokkosSparse_sptrsv_supernode.hpp"

namespace KokkosSparse { // kept because this file is temporary - see note above
namespace Experimental {


/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/* Auxiliary function for symbolic analysis                                                  */

/* ========================================================================================= */
template <typename graph_t, typename KernelHandle>
graph_t read_superlu_graphL(KernelHandle *kernelHandleL, SLU::SuperMatrix *L) {

  /* ---------------------------------------------------------------------- */
  /* get inputs */
  /* ---------------------------------------------------------------------- */
  int n = L->nrow;
  SLU::SCformat *Lstore = (SLU::SCformat*)(L->Store);

  int nsuper = 1 + Lstore->nsuper;     // # of supernodal columns
  int * mb = Lstore->rowind_colptr;
  int * nb = Lstore->sup_to_col;
  int * colptr = Lstore->nzval_colptr;
  int * rowind = Lstore->rowind;

  bool ptr_by_column = true;
  int nnzA = colptr[n] - colptr[0]; // overestimated if not block_diag
  return KokkosSparse::Experimental::read_supernodal_graphL<graph_t>
    (kernelHandleL, n, nsuper, nnzA, ptr_by_column, mb, nb, rowind);
}


/* ========================================================================================= */
// read SuperLU U factor into CSR
template <typename graph_t, typename KernelHandle>
graph_t read_superlu_graphU(KernelHandle kernelHandle, SLU::SuperMatrix *L,  SLU::SuperMatrix *U) {

  using   row_map_view_t = typename graph_t::row_map_type::non_const_type;
  using      cols_view_t = typename graph_t::entries_type::non_const_type;
  using host_cols_view_t = typename cols_view_t::HostMirror;
  using integer_view_host_t = Kokkos::View<int*, Kokkos::HostSpace>;

  /* load options */
  auto *handle = kernelHandle->get_sptrsv_handle ();
  bool u_in_csc = handle->is_column_major ();

  SLU::SCformat *Lstore = (SLU::SCformat*)(L->Store);
  SLU::NCformat *Ustore = (SLU::NCformat*)(U->Store);

  /* create a map from row id to supernode id */
  int n = L->nrow;
  int nsuper = 1 + Lstore->nsuper;     // # of supernodal columns
  int *nb = Lstore->sup_to_col;
  int *colptrU = Ustore->colptr;
  int *rowindU = Ustore->rowind;

  integer_view_host_t map ("map", n);
  int supid = 0;
  for (int k = 0; k < nsuper; k++) {
    int j1 = nb[k];
    int j2 = nb[k+1];
    for (int j = j1; j < j2; j++) {
      map (j) = supid;
    }
    supid ++;
  }

  /* count number of nonzeros in each row */
  row_map_view_t rowmap_view ("rowmap_view", n+1);
  auto hr = Kokkos::create_mirror_view (rowmap_view);
  Kokkos::deep_copy (hr, 0);

  integer_view_host_t sup ("sup", nsuper);
  integer_view_host_t check ("check", nsuper);
  Kokkos::deep_copy (check, 0);
  for (int k = nsuper-1; k >= 0; k--) {
    int j1 = nb[k];
    int nscol = nb[k+1] - j1;

    /* the diagonal block */
    for (int i = 0; i < nscol; i++) {
      hr (j1+i + 1) += nscol;
    }

    /* the off-diagonal blocks */
    // TODO: should take unions of nonzero columns per block row
    int nsup = 0;
    for (int jcol = j1; jcol < j1 + nscol; jcol++) {
      for (int i = colptrU[jcol]; i < colptrU[jcol+1]; i++ ){
        int irow = rowindU[i];
        supid = map (irow);
        if (check (supid) == 0) {
          if (u_in_csc) {
            int nsrow = nb[supid+1] - nb[supid];
            for (int jj = j1; jj < j1 + nscol; jj++) {
              hr (jj + 1) += nsrow;
            }
          } else {
            for (int ii = nb[supid]; ii < nb[supid+1]; ii++) {
              hr (ii + 1) += nscol;
            }
          }
          check (supid) = 1;
          sup (nsup) = supid;
          nsup ++;
        }
      }
    }
    // reset check
    for (int i = 0; i < nsup; i++ ) {
      check (sup (i)) = 0;
    }
  }

  // convert to the offset for each row
  for (int i = 1; i <= n; i++) {
    hr (i) += hr (i-1);
  }

  /* Upper-triangular matrix */
  auto nnzA = hr (n);
  cols_view_t column_view ("colmap_view", nnzA);
  host_cols_view_t hc = Kokkos::create_mirror_view (column_view);

  for (int k = 0; k < nsuper; k++) {
    int j1 = nb[k];
    int nscol = nb[k+1] - j1;

    /* the diagonal "dense" block */
    for (int i = 0; i < nscol; i++) {
      for (int j = 0; j < nscol; j++) {
        hc(hr(j1 + i) + j) = j1 + j;
      }
      hr (j1 + i) += nscol;
    }

    /* the off-diagonal "sparse" blocks */
    int nsup = 0;
    // let me first find off-diagonal supernodal blocks..
    for (int jcol = j1; jcol < j1 + nscol; jcol++) {
      for (int i = colptrU[jcol]; i < colptrU[jcol+1]; i++ ) {
        int irow = rowindU[i];
        if (check (map (irow)) == 0) {
          check (map (irow)) = 1;
          sup (nsup) = map (irow);
          nsup ++;
        }
      }
    }
    if (u_in_csc) {
      for (int jcol = j1; jcol < j1 + nscol; jcol++) {
        // move up all the row pointers for all the supernodal blocks
        for (int i = 0; i < nsup; i++) {
          for (int ii = nb[sup (i)]; ii < nb[sup (i) + 1]; ii++) {
            hc(hr(jcol)) = ii;
            hr(jcol) ++;
          }
        }
      }
    } else {
      for (int jcol = j1; jcol < j1 + nscol; jcol++) {
        // move up all the row pointers for all the supernodal blocks
        // (only nonzero columns)
        // TODO: should take unions of nonzero columns per block row
        for (int i = 0; i < nsup; i++) {
          for (int ii = nb[sup (i)]; ii < nb[sup (i) + 1]; ii++) {
            hc(hr(ii)) = jcol;
            hr(ii) ++;
          }
        }
      }
    }

    // reset check
    for (int i = 0; i < nsup; i++) {
      check (sup (i)) = 0;
    }
  }

  // fix hr
  for (int i = n; i >= 1; i--) {
    hr(i) = hr(i-1);
  }
  hr(0) = 0;
  #ifdef KOKKOS_SPTRSV_SUPERNODE_PROFILE
  std::cout << "    * Matrix size = " << n << std::endl;
  std::cout << "    * Total nnz   = " << hr (n) << std::endl;
  std::cout << "    * nnz / n     = " << hr (n)/n << std::endl;
  #endif

  // deepcopy
  Kokkos::deep_copy (rowmap_view, hr);
  Kokkos::deep_copy (column_view, hc);

  // create crsgraph
  graph_t static_graph (column_view, rowmap_view);
  return static_graph;
}


/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/* Symbolic analysis                                                                         */
template <typename KernelHandle>
void sptrsv_symbolic(
    KernelHandle *kernelHandleL,
    KernelHandle *kernelHandleU,
    SLU::SuperMatrix &L,
    SLU::SuperMatrix &U)
{
  using host_graph_t  = typename KernelHandle::SPTRSVHandleType::host_graph_t;

  Kokkos::Timer timer;
  Kokkos::Timer tic;
  timer.reset();

  // ===================================================================
  // load sptrsv-handles
  auto *handleL = kernelHandleL->get_sptrsv_handle ();
  int *etree = handleL->get_etree ();

  // ===================================================================
  // load options
  bool needEtree = (handleL->get_algorithm () == KokkosSparse::Experimental::SPTRSVAlgorithm::SUPERNODAL_SPMV ||
                    handleL->get_algorithm () == KokkosSparse::Experimental::SPTRSVAlgorithm::SUPERNODAL_ETREE);
  if (needEtree && etree == nullptr) {
    std::cout << std::endl
              << " ** etree needs to be set before calling sptrsv_symbolic with SuperLU **"
              << std::endl << std::endl;
    return;
  }

  // ===================================================================
  // read CrsGraph from SuperLU factor
  #ifdef KOKKOS_SPTRSV_SUPERNODE_PROFILE
  bool merge = handleL->get_merge_supernodes ();
  std::cout << " > Read SuperLU factor into KokkosSparse::CrsMatrix (invert diagonal and copy to device)" << std::endl;
  if (merge) {
    std::cout << " > Merge supernodes" << std::endl;
  }
  #endif
  host_graph_t graphL_host;
  host_graph_t graphU_host;

  tic.reset();
  graphL_host = read_superlu_graphL<host_graph_t> (kernelHandleL, &L);
  graphU_host = read_superlu_graphU<host_graph_t> (kernelHandleU, &L, &U);
  #ifdef KOKKOS_SPTRSV_SUPERNODE_PROFILE
  double time_seconds = tic.seconds ();
  std::cout << "   Conversion Time (from SuperLU to CSR): " << time_seconds << std::endl;
  #endif

  // ===================================================================
  // setup supnodal info
  SLU::SCformat *Lstore = (SLU::SCformat*)(L.Store);
  int nsuper = 1 + Lstore->nsuper;
  int *supercols = Lstore->sup_to_col;

  // ===================================================================
  // call supnodal symbolic
  KokkosSparse::Experimental::sptrsv_supernodal_symbolic (nsuper, supercols, etree,
                              graphL_host, kernelHandleL,
                              graphU_host, kernelHandleU);
}


/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/* Auxiliary functions for numeric computation                                               */

/* ========================================================================================= */
template <typename crsmat_t, typename graph_t, typename KernelHandle>
crsmat_t read_superlu_valuesL(KernelHandle kernelHandle, SLU::SuperMatrix *L, graph_t &static_graph) {

  using values_view_t = typename crsmat_t::values_type::non_const_type;
  using scalar_t      = typename values_view_t::value_type;

  /* ---------------------------------------------------------------------- */
  /* get inputs */
  /* ---------------------------------------------------------------------- */
  SLU::SCformat *Lstore = (SLU::SCformat*)(L->Store);
  scalar_t *Lx = (scalar_t*)(Lstore->nzval);

  int n = L->nrow;
  int nsuper = 1 + Lstore->nsuper;     // # of supernodal columns
  int * mb = Lstore->rowind_colptr;
  int * nb = Lstore->sup_to_col;
  int * colptr = Lstore->nzval_colptr;
  int * rowind = Lstore->rowind;

  bool unit_diag = true;
  bool ptr_by_column = true;
  return KokkosSparse::Experimental::read_supernodal_valuesL<crsmat_t>
    (unit_diag, kernelHandle, n, nsuper, ptr_by_column, mb, nb, colptr, rowind, Lx, static_graph);
}


/* ========================================================================================= */
// store numerical values of SuperLU U-factor into CSR
template <typename crsmat_t,
          typename graph_t,
          typename KernelHandle>
crsmat_t
read_superlu_valuesU(KernelHandle kernelHandle,
                     SLU::SuperMatrix *L,  SLU::SuperMatrix *U, graph_t &static_graph) {

  using values_view_t  = typename crsmat_t::values_type::non_const_type;
  using scalar_t       = typename values_view_t::value_type;
  using integer_view_host_t = Kokkos::View<int*, Kokkos::HostSpace>;

  const scalar_t zero (0.0);

  /* load options */
  // NOTE: invert-offdiag not supported in CSR format
  auto *handle = kernelHandle->get_sptrsv_handle ();
  bool u_in_csc = handle->is_column_major ();

  /* load inputs */
  SLU::SCformat *Lstore = (SLU::SCformat*)(L->Store);
  scalar_t *Lx = (scalar_t*)(Lstore->nzval);

  SLU::NCformat *Ustore = (SLU::NCformat*)(U->Store);
  scalar_t *Uval = (scalar_t*)(Ustore->nzval);

  int n = L->nrow;
  int nsuper = 1 + Lstore->nsuper;     // # of supernodal columns
  int *nb = Lstore->sup_to_col;
  int *mb = Lstore->rowind_colptr;
  int *colptrL = Lstore->nzval_colptr;
  int *colptrU = Ustore->colptr;
  int *rowindU = Ustore->rowind;

  /* create a map from row id to supernode id */
  int supid = 0;
  integer_view_host_t map ("map", n);
  for (int k = 0; k < nsuper; k++) {
    int j1 = nb[k];
    int j2 = nb[k+1];
    for (int j = j1; j < j2; j++) {
        map (j) = supid;
    }
    supid ++;
  }

  auto rowmap_view = static_graph.row_map;
  auto hr = Kokkos::create_mirror_view (rowmap_view);
  Kokkos::deep_copy (hr, rowmap_view);

  /* Upper-triangular matrix */
  auto nnzA = hr (n);
  values_view_t values_view ("values_view", nnzA);
  auto hv = Kokkos::create_mirror_view (values_view);
  Kokkos::deep_copy (hv, zero);

  integer_view_host_t sup ("supernodes", nsuper);
  integer_view_host_t off ("offsets", nsuper);
  integer_view_host_t check ("check", nsuper);
  Kokkos::deep_copy (check, 0);
  for (int k = 0; k < nsuper; k++) {
    int j1 = nb[k];
    int nscol = nb[k+1] - j1;

    int i1 = mb[j1];
    int nsrow = mb[j1+1] - i1;

    /* the diagonal "dense" block */
    int psx = colptrL[j1];
    if (u_in_csc) {
      for (int j = 0; j < nscol; j++) {
        for (int i = 0; i <= j; i++) {
          hv(hr(j1 + j) + i) = Lx[psx + i + j*nsrow];
        }
        hr (j1 + j) += nscol;
      }
    } else {
      for (int i = 0; i < nscol; i++) {
        for (int j = i; j < nscol; j++) {
          hv(hr(j1 + i) + j) = Lx[psx + i + j*nsrow];
        }
        hr (j1 + i) += nscol;
      }
    }

    /* the off-diagonal "sparse" blocks */
    // let me first find off-diagonal supernodal blocks..
    int nsup = 0;
    for (int jcol = j1; jcol < j1 + nscol; jcol++) {
      for (int i = colptrU[jcol]; i < colptrU[jcol+1]; i++ ){
        int irow = rowindU[i];
        if (check (map (irow)) == 0) {
          check (map (irow)) = 1;
          sup (nsup) = map (irow);
          nsup ++;
        }
      }
    }
    if (u_in_csc) {
      int offset = 0;
      for (int i = 0; i < nsup; i++) {
        off (sup (i)) = offset;
        offset += nb[sup (i) + 1] - nb[sup (i)];
      }
      for (int jcol = j1; jcol < j1 + nscol; jcol++) {
        // add nonzeros in jcol-th column
        for (int i = colptrU[jcol]; i < colptrU[jcol+1]; i++ ){
          int irow = rowindU[i];
          int id = map (irow);
          int ioff = off (id) + (irow - nb[id]);
          hv(hr(jcol) + ioff) = Uval[i];
        }
        // move up the pointers for all the supernodal blocks
        hr(jcol) += offset;
      }
    } else {
      for (int jcol = j1; jcol < j1 + nscol; jcol++) {
        // add nonzeros in jcol-th column
        // (only nonzero columns)
        // TODO: should take unions of nonzero columns per block row
        for (int i = colptrU[jcol]; i < colptrU[jcol+1]; i++ ){
          int irow = rowindU[i];
          hv(hr(irow)) = Uval[i];
        }
        // move up all the row pointers for all the supernodal blocks
        for (int i = 0; i < nsup; i++) {
          for (int ii = nb[sup (i)]; ii < nb[sup (i) + 1]; ii++) {
            hr(ii) ++;
          }
        }
      }
    }
    // reset check
    for (int i = 0; i < nsup; i++) {
      check (sup (i)) = 0;
    }
  }

  // fix hr
  for (int i = n; i >= 1; i--) {
    hr(i) = hr(i-1);
  }
  hr(0) = 0;
  #ifdef KOKKOS_SPTRSV_SUPERNODE_PROFILE
  std::cout << "    * Matrix size = " << n << std::endl;
  std::cout << "    * Total nnz   = " << hr (n) << std::endl;
  std::cout << "    * nnz / n     = " << hr (n)/n << std::endl;
  #endif

  // invert blocks (TODO: done on host for now)
  bool unit_diag = false;
  auto entries = static_graph.entries;
  auto hc = Kokkos::create_mirror_view (entries);
  Kokkos::deep_copy (hc, entries);
  KokkosSparse::Experimental::invert_supernodal_columns (kernelHandle, unit_diag, nsuper, nb, hr, hc, hv);
  // deepcopy
  Kokkos::deep_copy (values_view, hv);

  // create crs
  crsmat_t crsmat("CrsMatrix", n, values_view, static_graph);

  return crsmat;
}


/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/* For numeric computation                                                                   */
template <typename KernelHandle>
void sptrsv_compute(
    KernelHandle *kernelHandleL,
    KernelHandle *kernelHandleU,
    SLU::SuperMatrix &L,
    SLU::SuperMatrix &U)
{
  using      crsmat_t = typename KernelHandle::SPTRSVHandleType::crsmat_t;
  using host_crsmat_t = typename KernelHandle::SPTRSVHandleType::host_crsmat_t;

  Kokkos::Timer tic;
  Kokkos::Timer timer;
  // ===================================================================
  // load sptrsv-handles
  auto *handleL = kernelHandleL->get_sptrsv_handle ();
  auto *handleU = kernelHandleU->get_sptrsv_handle ();

  if (!(handleL->is_symbolic_complete()) ||
      !(handleU->is_symbolic_complete())) {
    std::cout << std::endl
            << " ** needs to call sptrsv_symbolic before calling sptrsv_numeric **"
           << std::endl << std::endl;
    return;
  }

  // ===================================================================
  // load options
  bool merge = handleL->get_merge_supernodes ();
  bool useSpMV = (handleL->get_algorithm () == KokkosSparse::Experimental::SPTRSVAlgorithm::SUPERNODAL_SPMV ||
                  handleL->get_algorithm () == KokkosSparse::Experimental::SPTRSVAlgorithm::SUPERNODAL_SPMV_DAG);
  #ifdef KOKKOS_SPTRSV_SUPERNODE_PROFILE
  double time_seconds = 0.0;
  bool invert_offdiag = handleL->get_invert_offdiagonal ();
  if (merge)          std::cout << " >> merge\n" << std::endl;
  if (invert_offdiag) std::cout << " >> invert offdiag\n" << std::endl;
  #endif

  // ===================================================================
  // load graphs
  auto graphL = handleL->get_graph ();
  auto graphL_host = handleL->get_graph_host ();

  auto graphU = handleU->get_graph ();
  auto graphU_host = handleU->get_graph_host ();

  int nsuper = handleL->get_num_supernodes ();
  const int* supercols = handleL->get_supercols_host ();
  crsmat_t superluL, superluU;
  host_crsmat_t superluL_host, superluU_host;
  if (merge) {
    tic.reset ();
    // ========================================================
    // read in the numerical L-values into merged csc
    // NOTE: we first load into CRS, and then merge (should be combined)
    // 1) load L into crs (offdiagonal not inverted, unless invert diag)
    bool invert_diag = handleL->get_invert_diagonal ();
    kernelHandleL->set_sptrsv_invert_diagonal (false);  // invert after merge
    auto original_graphL_host = handleL->get_original_graph_host ();
    superluL_host = read_superlu_valuesL<host_crsmat_t> (kernelHandleL, &L, original_graphL_host);
    // 2) re-load L into merged crs
    bool unit_diag = true;
    kernelHandleL->set_sptrsv_invert_diagonal (invert_diag);  // now invert, TODO: diagonals are always inverted
    if (useSpMV) {
      superluL_host = KokkosSparse::Experimental::read_merged_supernodes<host_crsmat_t> (kernelHandleL, nsuper, supercols,
                                                             unit_diag, superluL_host, graphL_host);
    } else {
      superluL = KokkosSparse::Experimental::read_merged_supernodes<crsmat_t> (kernelHandleL, nsuper, supercols,
                                                   unit_diag, superluL_host, graphL);
    }

    // ========================================================
    // read in the numerical U-values into merged csr
    // 1) load U into crs
    invert_diag = handleU->get_invert_diagonal ();
    kernelHandleU->set_sptrsv_invert_diagonal (false);  // invert after merge
    auto original_graphU_host = handleU->get_original_graph_host ();
    superluU_host = read_superlu_valuesU<host_crsmat_t> (kernelHandleU, &L, &U, original_graphU_host);
    // 2) re-load U into merged crs
    unit_diag = false;
    kernelHandleU->set_sptrsv_invert_diagonal (invert_diag);  // now invert, TODO: diagonals are always inverted
    if (useSpMV) {
      superluU_host = KokkosSparse::Experimental::read_merged_supernodes<host_crsmat_t> (kernelHandleU, nsuper, supercols,
                                                             unit_diag, superluU_host, graphU_host);
    } else {
      superluU = KokkosSparse::Experimental::read_merged_supernodes<crsmat_t> (kernelHandleU, nsuper, supercols,
                                                   unit_diag, superluU_host, graphU);
    }
    #ifdef KOKKOS_SPTRSV_SUPERNODE_PROFILE
    time_seconds = tic.seconds ();
    std::cout << "   Time to Merge and Copy to device: " << time_seconds << std::endl;
    #endif
  } else {
    // ========================================================
    // read in the numerical values into merged csc for L
    //kernelHandleL->set_sptrsv_invert_diagonal (true); // only, invert diag is supported for now
    tic.reset ();
    if (useSpMV) {
      superluL_host = read_superlu_valuesL<host_crsmat_t> (kernelHandleL, &L, graphL_host);
    } else {
      superluL = read_superlu_valuesL<crsmat_t> (kernelHandleL, &L, graphL);
    }
    #ifdef KOKKOS_SPTRSV_SUPERNODE_PROFILE
    double timeL = tic.seconds ();
    #endif

    // ========================================================
    // read in the numerical values into merged csc/csr for U
    tic.reset ();
    //kernelHandleU->set_sptrsv_invert_diagonal (true); // only, invert diag is supported for now
    if (useSpMV) {
      superluU_host = read_superlu_valuesU<host_crsmat_t> (kernelHandleU, &L, &U, graphU_host);
    } else {
      superluU = read_superlu_valuesU<crsmat_t> (kernelHandleU, &L, &U, graphU);
    }
    #ifdef KOKKOS_SPTRSV_SUPERNODE_PROFILE
    double timeU = tic.seconds ();
    std::cout << "   Time to copy to device: " << std::endl;
    std::cout << "   > copy L to device: " << timeL << std::endl;
    std::cout << "   > copy U to device: " << timeU << std::endl;
    #endif
  }

  // ===================================================================
  if (useSpMV) {
    // ----------------------------------------------------
    // split the matrix into submatrices for spmv at each level
    tic.reset ();
    KokkosSparse::Experimental::split_crsmat<crsmat_t> (kernelHandleL, superluL_host);
    KokkosSparse::Experimental::split_crsmat<crsmat_t> (kernelHandleU, superluU_host);
    #ifdef KOKKOS_SPTRSV_SUPERNODE_PROFILE
    time_seconds = tic.seconds ();
    std::cout << "   Time to Split to submatrix: " << time_seconds << std::endl;
    #endif
  }


  // ==============================================
  // save crsmat
  handleL->set_crsmat (superluL);
  handleU->set_crsmat (superluU);

  // ===================================================================
  handleL->set_numeric_complete ();
  handleU->set_numeric_complete ();
  #ifdef KOKKOS_SPTRSV_SUPERNODE_PROFILE
  time_seconds = timer.seconds ();
  std::cout << "   Total Compute Time: " << time_seconds << std::endl << std::endl;
  #endif
} // sptrsv_compute

} // end namespace EXPERIMENTAL
} // end namespace Amesos2

#endif // KOKKOSKERNELS_ENABLE_TPL_SUPERLU
#endif // AMESOS2_SUPERLU_TRSV_HPP_

