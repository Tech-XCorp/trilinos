/*
// @HEADER
// ***********************************************************************
//
//          Tpetra: Templated Linear Algebra Services Package
//                 Copyright (2008) Sandia Corporation
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
// ************************************************************************
// @HEADER
*/

#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Details_gathervPrint.hpp>
#include <stddef.h>
#include <algorithm>
#include <string>
#include <type_traits>

#include "Teuchos_ArrayRCP.hpp"
#include "Teuchos_ArrayRCPDecl.hpp"
#include "Teuchos_Comm.hpp"
#include "Teuchos_Ptr.hpp"
#include "Teuchos_PtrDecl.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_RCPDecl.hpp"
#include "Teuchos_RCPNode.hpp"

namespace Tpetra {
namespace Details {

void
gathervPrint (std::ostream& out,
              const std::string& s,
              const Teuchos::Comm<int>& comm)
{
  using Teuchos::ArrayRCP;
  using Teuchos::CommRequest;
  using Teuchos::ireceive;
  using Teuchos::isend;
  using Teuchos::outArg;
  using Teuchos::RCP;
  using Teuchos::wait;

  const int myRank = comm.getRank ();
  const int rootRank = 0;
  if (myRank == rootRank) {
    out << s; // Proc 0 prints its buffer first
  }

  const int numProcs = comm.getSize ();
  const int sizeTag = 42;
  const int msgTag = 43;

  ArrayRCP<size_t> sizeBuf (1);
  ArrayRCP<char> msgBuf; // to be resized later
  RCP<CommRequest<int> > req;

  for (int p = 1; p < numProcs; ++p) {
    if (myRank == p) {
      sizeBuf[0] = s.size ();
      req = isend<int, size_t> (sizeBuf, rootRank, sizeTag, comm);
      (void) wait<int> (comm, outArg (req));

      const size_t msgSize = s.size ();
      msgBuf.resize (msgSize + 1); // for the '\0'
      std::copy (s.begin (), s.end (), msgBuf.begin ());
      msgBuf[msgSize] = '\0';

      req = isend<int, char> (msgBuf, rootRank, msgTag, comm);
      (void) wait<int> (comm, outArg (req));
    }
    else if (myRank == rootRank) {
      sizeBuf[0] = 0; // just a precaution
      req = ireceive<int, size_t> (sizeBuf, p, sizeTag, comm);
      (void) wait<int> (comm, outArg (req));

      const size_t msgSize = sizeBuf[0];
      msgBuf.resize (msgSize + 1); // for the '\0'
      req = ireceive<int, char> (msgBuf, p, msgTag, comm);
      (void) wait<int> (comm, outArg (req));

      std::string msg (msgBuf.getRawPtr ());
      out << msg;
    }
  }
}

} // namespace Details
} // namespace Tpetra
