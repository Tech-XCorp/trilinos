// @HEADER
//
// ***********************************************************************
//
//             Xpetra: A linear algebra interface package
//                  Copyright 2012 Sandia Corporation
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
// Questions? Contact
//                    Jonathan Hu       (jhu@sandia.gov)
//                    Andrey Prokopenko (aprokop@sandia.gov)
//                    Ray Tuminaro      (rstumin@sandia.gov)
//
// ***********************************************************************
//
// @HEADER
#ifndef XPETRA_COMM_HPP
#define XPETRA_COMM_HPP

//! Conversion between Epetra and Teuchos objects

#include "Teuchos_RCPDecl.hpp"
#include "Xpetra_ConfigDefs.hpp"
#include "Xpetra_config.hpp"

class Epetra_Comm;
namespace Teuchos {
template <typename Ordinal> class Comm;
}  // namespace Teuchos

#ifdef HAVE_XPETRA_EPETRA

#include <Epetra_Comm.h>
// header file for Teuchos::ETransp
#include <Teuchos_BLAS_types.hpp>
// header files for comm objects conversion
#include <Teuchos_Comm.hpp>

namespace Xpetra {

  using Teuchos::RCP;

  //! Convert a Teuchos_Comm to an Epetra_Comm.
  const RCP<const Epetra_Comm> toEpetra(const RCP<const Teuchos::Comm<int> > & comm);

  //! Convert an Epetra_Comm.to a Teuchos_Comm.
  const RCP<const Teuchos::Comm<int> > toXpetra(const Epetra_Comm & comm);

  //! Convert a Teuchos::ETransp to an Epetra boolean.
  bool toEpetra(Teuchos::ETransp);

}
#endif // HAVE_XPETRA_EPETRA

#endif // XPETRA_EPETRACOMM_HPP

// Note: no support for Epetra_MpiSmpComm
// TODO: remove return RCP for toEpetra?
