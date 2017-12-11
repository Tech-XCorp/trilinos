//@HEADER
// ************************************************************************
//
//                 Belos: Block Linear Solvers Package
//                  Copyright 2004 Sandia Corporation
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
//@HEADER


#include "BelosSolverFactory_decl.hpp"
#include "BelosSolverFactory_impl.hpp"
#include "BelosOperator.hpp"
#include "BelosMultiVec.hpp"


// TODO run this for enable Tpetra only
// Add #include statements here to pull in the code that will be
// needed to generate one or more validated ParameterLists
#include "Tpetra_MultiVector.hpp"
#include "BelosTpetraAdapter.hpp"
#include "BelosSolverFactory.hpp"

typedef Tpetra::MultiVector<> multivector_type;
typedef Tpetra::Operator<> operator_type;
typedef multivector_type::scalar_type scalar_type;
typedef Belos::SolverManager<scalar_type, multivector_type, operator_type> solver_type;

namespace Belos {
	Belos::SolverFactory<scalar_type, multivector_type, operator_type> sFactory;
	Teuchos::RCP<solver_type> solver;
  
};

// This covers src/test/Factory.cpp
namespace Belos {
  typedef double ST;
  typedef Belos::MultiVec<ST> MV;
  typedef Belos::Operator<ST> OP;
  typedef Belos::SolverManager<ST, MV, OP> solver_base_type;
  typedef Belos::SolverFactory<ST, MV, OP> factory_type;
  template class SolverFactory<ST, Belos::MultiVec<ST>, Belos::Operator<ST>>;
  template class SolverManager<ST, Belos::MultiVec<ST>, Belos::Operator<ST>>;

} // namespace Belos

