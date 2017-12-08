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

#ifndef __Belos_SolverFactory_impl_hpp
#define __Belos_SolverFactory_impl_hpp

// TODO: Move includes from decl as appropriate

namespace Belos {
namespace Details {

/// \fn makeSolverManagerFromEnum
/// \brief Return a new instance of the desired SolverManager subclass.
/// \author Mark Hoemmen
///
/// The \c SolverFactory class may use this template function
/// in order to instantiate an instance of the desired subclass of \c
/// SolverManager.
///
/// \tparam Scalar The first template parameter of \c SolverManager.
/// \tparam MV The second template parameter of \c SolverManager.
/// \tparam OP The third template parameter of \c SolverManager.
///
/// \param solverType [in] Enum value representing the specific
///   SolverManager subclass to instantiate.
///
/// \param params [in/out] List of parameters with which to configure
///   the solver.  If null, we configure the solver with default
///   parameters.
template<class Scalar, class MV, class OP>
Teuchos::RCP<SolverManager<Scalar, MV, OP> >
makeSolverManagerFromEnum (const EBelosSolverType solverType,
                           const Teuchos::RCP<Teuchos::ParameterList>& params)
{
  typedef SolverManager<Scalar, MV, OP> base_type;

  switch (solverType) {
  case SOLVER_TYPE_BLOCK_GMRES: {
    typedef BlockGmresSolMgr<Scalar, MV, OP> impl_type;
    return makeSolverManagerTmpl<base_type, impl_type> (params);
  }
  case SOLVER_TYPE_PSEUDO_BLOCK_GMRES: {
    typedef PseudoBlockGmresSolMgr<Scalar, MV, OP> impl_type;
    return makeSolverManagerTmpl<base_type, impl_type> (params);
  }
  case SOLVER_TYPE_BLOCK_CG: {
    typedef BlockCGSolMgr<Scalar, MV, OP> impl_type;
    return makeSolverManagerTmpl<base_type, impl_type> (params);
  }
  case SOLVER_TYPE_PSEUDO_BLOCK_CG: {
    typedef PseudoBlockCGSolMgr<Scalar, MV, OP> impl_type;
    return makeSolverManagerTmpl<base_type, impl_type> (params);
  }
  case SOLVER_TYPE_GCRODR: {
    typedef GCRODRSolMgr<Scalar, MV, OP> impl_type;
    return makeSolverManagerTmpl<base_type, impl_type> (params);
  }
  case SOLVER_TYPE_RCG: {
    typedef RCGSolMgr<Scalar, MV, OP> impl_type;
    return makeSolverManagerTmpl<base_type, impl_type> (params);
  }
  case SOLVER_TYPE_MINRES: {
    typedef MinresSolMgr<Scalar, MV, OP> impl_type;
    return makeSolverManagerTmpl<base_type, impl_type> (params);
  }
  case SOLVER_TYPE_LSQR: {
    typedef LSQRSolMgr<Scalar, MV, OP> impl_type;
    return makeSolverManagerTmpl<base_type, impl_type> (params);
  }
  case SOLVER_TYPE_STOCHASTIC_CG: {
    typedef PseudoBlockStochasticCGSolMgr<Scalar, MV, OP> impl_type;
    return makeSolverManagerTmpl<base_type, impl_type> (params);
  }
  case SOLVER_TYPE_TFQMR: {
    typedef TFQMRSolMgr<Scalar, MV, OP> impl_type;
    return makeSolverManagerTmpl<base_type, impl_type> (params);
  }
  case SOLVER_TYPE_PSEUDO_BLOCK_TFQMR: {
    typedef PseudoBlockTFQMRSolMgr<Scalar, MV, OP> impl_type;
    return makeSolverManagerTmpl<base_type, impl_type> (params);
  }
  case SOLVER_TYPE_GMRES_POLY: {
    typedef GmresPolySolMgr<Scalar, MV, OP> impl_type;
    return makeSolverManagerTmpl<base_type, impl_type> (params);
  }
  case SOLVER_TYPE_PCPG: {
    typedef PCPGSolMgr<Scalar, MV, OP> impl_type;
    return makeSolverManagerTmpl<base_type, impl_type> (params);
  }
  case SOLVER_TYPE_FIXED_POINT: {
    typedef FixedPointSolMgr<Scalar, MV, OP> impl_type;
    return makeSolverManagerTmpl<base_type, impl_type> (params);
  }
  case SOLVER_TYPE_BICGSTAB: {
    typedef BiCGStabSolMgr<Scalar, MV, OP> impl_type;
    return makeSolverManagerTmpl<base_type, impl_type> (params);
  }
  default: // Fall through; let the code below handle it.
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, std::logic_error, "Belos::SolverFactory: Invalid EBelosSolverType "
      "enum value " << solverType << ".  Please report this bug to the Belos "
      "developers.");
  }

  // Compiler guard.  This may result in a warning on some compilers
  // for an unreachable statement, but it will prevent a warning on
  // other compilers for a "missing return statement at end of
  // non-void function."
  TEUCHOS_UNREACHABLE_RETURN(Teuchos::null);
}

template<class SolverManagerBaseType, class SolverManagerType>
Teuchos::RCP<SolverManagerBaseType>
makeSolverManagerTmpl (const Teuchos::RCP<Teuchos::ParameterList>& params)
{
  using Teuchos::ParameterList;
  using Teuchos::parameterList;
  using Teuchos::RCP;

  RCP<SolverManagerType> solver = rcp (new SolverManagerType);

  // Some solvers may not like to get a null ParameterList.  If params
  // is null, replace it with an empty parameter list.  The solver
  // will fill in default parameters for that case.  Use the name of
  // the solver's default parameters to name the new empty list.
  RCP<ParameterList> pl;
  if (params.is_null()) {
    pl = parameterList (solver->getValidParameters ()->name ());
  } else {
    pl = params;
  }
  TEUCHOS_TEST_FOR_EXCEPTION(
    pl.is_null(), std::logic_error,
    "Belos::SolverFactory: ParameterList to pass to solver is null.  This "
    "should never happen.  Please report this bug to the Belos developers.");
  solver->setParameters (pl);
  return solver;
}

} // namespace Details

namespace Impl {

template<class Scalar, class MV, class OP>
void
SolverFactoryParent<Scalar, MV, OP>::
addFactory (const Teuchos::RCP<CustomSolverFactory<Scalar, MV, OP> >& factory)
{
  factories_.push_back (factory);
}


template<class Scalar, class MV, class OP>
std::string
SolverFactoryParent<Scalar, MV, OP>::
description () const
{
  using Teuchos::TypeNameTraits;

  std::ostringstream out;
  out << "\"Belos::SolverFactory\": {";
  if (this->getObjectLabel () != "") {
    out << "Label: " << this->getObjectLabel () << ", ";
  }
  out << "Scalar: \"" << TypeNameTraits<Scalar>::name ()
      << "\", MV: \"" << TypeNameTraits<MV>::name ()
      << "\", OP: \"" << TypeNameTraits<OP>::name ()
      << "\"}";
  return out.str ();
}


template<class Scalar, class MV, class OP>
void
SolverFactoryParent<Scalar, MV, OP>::
describe (Teuchos::FancyOStream& out,
          const Teuchos::EVerbosityLevel verbLevel) const
{
  using Teuchos::TypeNameTraits;
  using std::endl;

  const Teuchos::EVerbosityLevel vl =
    (verbLevel == Teuchos::VERB_DEFAULT) ? Teuchos::VERB_LOW : verbLevel;

  if (vl == Teuchos::VERB_NONE) {
    return;
  }

  // By convention, describe() always begins with a tab.
  Teuchos::OSTab tab0 (out);
  // The description prints in YAML format.  The class name needs to
  // be protected with quotes, so that YAML doesn't get confused
  // between the colons in the class name and the colon separating
  // (key,value) pairs.
  out << "\"Belos::SolverFactory\":" << endl;
  if (this->getObjectLabel () != "") {
    out << "Label: " << this->getObjectLabel () << endl;
  }
  {
    out << "Template parameters:" << endl;
    Teuchos::OSTab tab1 (out);
    out << "Scalar: \"" << TypeNameTraits<Scalar>::name () << "\"" << endl
        << "MV: \"" << TypeNameTraits<MV>::name () << "\"" << endl
        << "OP: \"" << TypeNameTraits<OP>::name () << "\"" << endl;
  }

  // At higher verbosity levels, print out the list of supported solvers.
  if (vl > Teuchos::VERB_LOW) {
    Teuchos::OSTab tab1 (out);
    out << "Number of solvers: " << numSupportedSolvers ()
        << endl;
    out << "Canonical solver names: ";
    Impl::printStringArray (out, Details::canonicalSolverNames ());
    out << endl;

    out << "Aliases to canonical names: ";
    Impl::printStringArray (out, Details::solverNameAliases ());
    out << endl;
  }
}

template<class Scalar, class MV, class OP>
int
SolverFactoryParent<Scalar, MV, OP>::
numSupportedSolvers () const
{
  int numSupported = 0;

  // First, check the overriding factories.
  for (std::size_t k = 0; k < factories_.size (); ++k) {
    using Teuchos::RCP;
    RCP<custom_solver_factory_type> factory = factories_[k];
    if (! factory.is_null ()) {
      numSupported += factory->numSupportedSolvers ();
    }
  }

  // Now, see how many solvers this factory supports.
  return numSupported + Details::numSupportedSolvers ();
}

template<class Scalar, class MV, class OP>
Teuchos::Array<std::string>
SolverFactoryParent<Scalar, MV, OP>::
supportedSolverNames () const
{
  typedef std::vector<std::string>::const_iterator iter_type;
  Teuchos::Array<std::string> names;

  // First, check the overriding factories.
  const std::size_t numFactories = factories_.size ();
  for (std::size_t factInd = 0; factInd < numFactories; ++factInd) {
    Teuchos::RCP<custom_solver_factory_type> factory = factories_[factInd];
    if (! factory.is_null ()) {
      std::vector<std::string> supportedSolvers =
        factory->supportedSolverNames ();
      const std::size_t numSolvers = supportedSolvers.size ();
      for (std::size_t solvInd = 0; solvInd < numSolvers; ++solvInd) {
        names.push_back (supportedSolvers[solvInd]);
      }
    }
  }

  {
    std::vector<std::string> aliases = Details::solverNameAliases ();
    for (iter_type iter = aliases.begin (); iter != aliases.end (); ++iter) {
      names.push_back (*iter);
    }
  }
  {
    std::vector<std::string> canonicalNames = Details::canonicalSolverNames ();
    for (iter_type iter = canonicalNames.begin ();
         iter != canonicalNames.end (); ++iter) {
      names.push_back (*iter);
    }
  }
  return names;
}

template<class Scalar, class MV, class OP>
bool
SolverFactoryParent<Scalar, MV, OP>::
isSupported (const std::string& solverName) const
{
  // First, check the overriding factories.
  const std::size_t numFactories = factories_.size ();
  for (std::size_t factInd = 0; factInd < numFactories; ++factInd) {
    using Teuchos::RCP;
    RCP<custom_solver_factory_type> factory = factories_[factInd];
    if (! factory.is_null ()) {
      if (factory->isSupported (solverName)) {
        return true;
      }
    }
  }
  // Now, check this factory.

  // Upper-case version of the input solver name.
  const std::string solverNameUC = Impl::upperCase (solverName);

  // Check whether the given name is an alias.
  std::pair<std::string, bool> aliasResult =
    Details::getCanonicalNameFromAlias (solverNameUC);
  const std::string candidateCanonicalName = aliasResult.first;
  const bool isAnAlias = aliasResult.second;

  // Get the canonical name.
  const Details::EBelosSolverType solverEnum =
    Details::getEnumFromCanonicalName (isAnAlias ?
                                       candidateCanonicalName :
                                       solverNameUC);
  const bool validCanonicalName =
    (solverEnum != Details::SOLVER_TYPE_UPPER_BOUND);
  return validCanonicalName;
}

} // namespace Impl
} // namespace Belos

#endif // __Belos_SolverFactory_impl_hpp

