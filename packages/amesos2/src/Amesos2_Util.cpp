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

#include "Amesos2_Util.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_DefaultSerialComm.hpp>

#ifdef HAVE_MPI
#  include <Teuchos_DefaultMpiComm.hpp>
#  ifdef HAVE_AMESOS2_EPETRA
#    include <Teuchos_OpaqueWrapper.hpp>
#  endif // HAVE_AMESOS2_EPETRA
#endif // HAVE_MPI

#ifdef HAVE_AMESOS2_EPETRA
#  include <Epetra_Map.h>
#  ifdef HAVE_MPI
#    include <Epetra_MpiComm.h>
#  endif // HAVE_MPI
#  include <Epetra_SerialComm.h>
#endif // HAVE_AMESOS2_EPETRA

#ifdef HAVE_AMESOS2_METIS
#include "metis.h"
#endif

#ifdef HAVE_AMESOS2_METIS
void
Amesos2::Util::metis_resort(host_metis_array & row_ptr, host_metis_array & cols,
  host_metis_array & perm, host_metis_array & peri) {
  // idx_t is the type metis was built with and the input arrays will be type
  // int64_t, which should match idx_t, meaning no actual copies here.
  // In case metis is using int32_t, copies here will convert it.
  typedef Kokkos::View<idx_t*, Kokkos::HostSpace>  metis_view_t;
  metis_view_t metis_row_ptr;
  metis_view_t metis_cols;
  deep_copy_or_assign_view(metis_row_ptr, row_ptr);
  deep_copy_or_assign_view(metis_cols, cols);
  metis_view_t metis_perm(Kokkos::ViewAllocateWithoutInitializing("metis_perm"), perm.size());
  metis_view_t metis_peri(Kokkos::ViewAllocateWithoutInitializing("metis_peri"), peri.size());
  idx_t metis_size;
  int err = METIS_NodeND(&metis_size, metis_row_ptr.data(), metis_cols.data(),
    NULL, NULL, metis_perm.data(), metis_peri.data());
  TEUCHOS_TEST_FOR_EXCEPTION(err != METIS_OK, std::runtime_error,
    "METIS_NodeND failed to sort matrix.");
  deep_copy_or_assign_view(perm, metis_perm);
  deep_copy_or_assign_view(peri, metis_peri);
}
#endif

#ifdef HAVE_AMESOS2_EPETRA
const Teuchos::RCP<const Teuchos::Comm<int> >
Amesos2::Util::to_teuchos_comm(Teuchos::RCP<const Epetra_Comm> c)
{
  using Teuchos::rcp;
  using Teuchos::rcp_dynamic_cast;
  using Teuchos::set_extra_data;

#ifdef HAVE_MPI
  Teuchos::RCP<const Epetra_MpiComm>
    mpiEpetraComm = rcp_dynamic_cast<const Epetra_MpiComm>(c);
  if( mpiEpetraComm.get() ) {
    Teuchos::RCP<const Teuchos::OpaqueWrapper<MPI_Comm> >
      rawMpiComm = Teuchos::opaqueWrapper(mpiEpetraComm->Comm());
    return Teuchos::createMpiComm<int>(rawMpiComm);
  } else
#endif // HAVE_MPI
    if( rcp_dynamic_cast<const Epetra_SerialComm>(c) != Teuchos::null )
      return Teuchos::createSerialComm<int>();
    else
      return(Teuchos::null);
}

const Teuchos::RCP<const Epetra_Comm>
Amesos2::Util::to_epetra_comm(Teuchos::RCP<const Teuchos::Comm<int> > c)
{
  using Teuchos::rcp;
  using Teuchos::rcp_dynamic_cast;
  using Teuchos::set_extra_data;

#ifdef HAVE_MPI
  Teuchos::RCP<const Teuchos::MpiComm<int> >
    mpiTeuchosComm = rcp_dynamic_cast<const Teuchos::MpiComm<int> >(c);
  if( mpiTeuchosComm.get() ){
    Teuchos::RCP<const Teuchos::OpaqueWrapper<MPI_Comm> >
      rawMpiComm = mpiTeuchosComm->getRawMpiComm();
    Teuchos::RCP<const Epetra_MpiComm>
      mpiComm = rcp(new Epetra_MpiComm(*rawMpiComm()));
    return mpiComm;
  }
#else // NOT HAVE_MPI
  Teuchos::RCP<const Teuchos::SerialComm<int> >
    serialTeuchosComm = rcp_dynamic_cast<const Teuchos::SerialComm<int> >(c);

  if( serialTeuchosComm.get() ){
    Teuchos::RCP<const Epetra_SerialComm> serialComm = rcp(new Epetra_SerialComm());
    return serialComm;
  }
#endif // HAVE_MPI

  return Teuchos::null;
}
#endif // HAVE_AMESOS2_EPETRA

/// Prints a line of 80 "-"s on out.
void Amesos2::Util::printLine( Teuchos::FancyOStream& out )
{
  out << "----------------------------------------"
      << "----------------------------------------"
      << std::endl;
}


