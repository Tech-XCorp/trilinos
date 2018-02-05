//@HEADER
// ***********************************************************************
//
//     EpetraExt: Epetra Extended - Linear Algebra Services Package
//                 Copyright (2011) Sandia Corporation
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
//@HEADER

#include <EpetraExt_Reindex_MultiVector.h>
#include <Epetra_Map.h>
#include <Epetra_MultiVector.h>
#include <assert.h>
#include <vector>

#include "EpetraExt_Transform.h"
#include "Epetra_DataAccess.h"
#include "Teuchos_RCP.hpp"

namespace EpetraExt {

MultiVector_Reindex::
~MultiVector_Reindex()
{
  if( newObj_ ) delete newObj_;
}

MultiVector_Reindex::NewTypeRef
MultiVector_Reindex::
operator()( OriginalTypeRef orig )
{
  origObj_ = &orig;

  //test std::map, must have same number of local and global elements as original row std::map
  assert( orig.Map().NumMyElements() == NewRowMap_.NumMyElements() );

  std::vector<double*> MyValues(1);
  int MyLDA;
  int NumVectors = orig.NumVectors();
  orig.ExtractView( &MyValues[0], &MyLDA );

  Epetra_MultiVector * NewMV = new Epetra_MultiVector( View, NewRowMap_, MyValues[0], MyLDA, NumVectors );

  newObj_ = NewMV;

  return *NewMV;
}

MultiVector_Reindex::NewTypeRCP
MultiVector_Reindex::transform(OriginalTypeRef orig)
{
  //test std::map, must have same number of local and global elements as original row std::map
  assert( orig.Map().NumMyElements() == NewRowMap_.NumMyElements() );

  std::vector<double*> MyValues(1);
  int MyLDA;
  int NumVectors = orig.NumVectors();
  orig.ExtractView( &MyValues[0], &MyLDA );

  return Teuchos::rcp(new Epetra_MultiVector( View, NewRowMap_, MyValues[0], MyLDA, NumVectors ));
}

} // namespace EpetraExt

