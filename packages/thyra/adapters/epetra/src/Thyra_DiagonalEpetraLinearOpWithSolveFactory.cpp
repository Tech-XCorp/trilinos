// @HEADER
// ***********************************************************************
// 
//    Thyra: Interfaces and Support for Abstract Numerical Algorithms
//                 Copyright (2004) Sandia Corporation
// 
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
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
// Questions? Contact Roscoe A. Bartlett (bartlettra@ornl.gov) 
// 
// ***********************************************************************
// @HEADER

#include <stddef.h>
#include <ostream>
#include <stdexcept>
#include <type_traits>

#include "Epetra_Map.h"
#include "Epetra_Operator.h"
#include "Epetra_RowMatrix.h"
#include "Epetra_Vector.h"
#include "Teuchos_ENull.hpp"
#include "Teuchos_Ptr.hpp"
#include "Teuchos_PtrDecl.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_RCPNode.hpp"
#include "Teuchos_TestForException.hpp"
#include "Teuchos_dyn_cast.hpp"
#include "Thyra_DefaultDiagonalLinearOpWithSolve_decl.hpp"
#include "Thyra_DiagonalEpetraLinearOpWithSolveFactory.hpp"
#include "Thyra_EpetraLinearOpBase.hpp"
#include "Thyra_EpetraThyraWrappers.hpp"
#include "Thyra_EpetraTypes.hpp"
#include "Thyra_LinearOpBase_decl.hpp"
#include "Thyra_LinearOpSourceBase.hpp"
#include "Thyra_LinearOpWithSolveBase_decl.hpp"

namespace Teuchos {
class ParameterList;
}  // namespace Teuchos


namespace Thyra {


template <class Scalar> class PreconditionerBase;
template <class Scalar> class VectorBase;
template <class Scalar> class VectorSpaceBase;

bool DiagonalEpetraLinearOpWithSolveFactory::isCompatible(
  const LinearOpSourceBase<double> &fwdOpSrc
  ) const
{
  using Teuchos::outArg;
  RCP<const LinearOpBase<double> >
    fwdOp = fwdOpSrc.getOp();
  const EpetraLinearOpBase *eFwdOp = NULL;
  if( ! (eFwdOp = dynamic_cast<const EpetraLinearOpBase*>(&*fwdOp)) )
    return false;
  RCP<const Epetra_Operator> epetraFwdOp;
  EOpTransp epetraFwdOpTransp;
  EApplyEpetraOpAs epetraFwdOpApplyAs;
  EAdjointEpetraOp epetraFwdOpAdjointSupport;
  eFwdOp->getEpetraOpView(outArg(epetraFwdOp), outArg(epetraFwdOpTransp),
    outArg(epetraFwdOpApplyAs), outArg(epetraFwdOpAdjointSupport) );
  if( !dynamic_cast<const Epetra_RowMatrix*>(&*epetraFwdOp) )
    return false;
  return true;
}


RCP<LinearOpWithSolveBase<double> >
DiagonalEpetraLinearOpWithSolveFactory::createOp() const
{
  return Teuchos::rcp(new DefaultDiagonalLinearOpWithSolve<double>());
}


void DiagonalEpetraLinearOpWithSolveFactory::initializeOp(
  const RCP<const LinearOpSourceBase<double> >    &fwdOpSrc
  ,LinearOpWithSolveBase<double>                                   *Op
  ,const ESupportSolveUse                                          supportSolveUse
  ) const
{
  using Teuchos::outArg;
  TEUCHOS_TEST_FOR_EXCEPT(Op==NULL);
  TEUCHOS_TEST_FOR_EXCEPT(fwdOpSrc.get()==NULL);
  TEUCHOS_TEST_FOR_EXCEPT(fwdOpSrc->getOp().get()==NULL);
  RCP<const LinearOpBase<double> > fwdOp = fwdOpSrc->getOp();
  const EpetraLinearOpBase &eFwdOp = Teuchos::dyn_cast<const EpetraLinearOpBase>(*fwdOp);
  RCP<const Epetra_Operator> epetraFwdOp;
  EOpTransp epetraFwdOpTransp;
  EApplyEpetraOpAs epetraFwdOpApplyAs;
  EAdjointEpetraOp epetraFwdOpAdjointSupport;
  eFwdOp.getEpetraOpView(outArg(epetraFwdOp), outArg(epetraFwdOpTransp),
    outArg(epetraFwdOpApplyAs), outArg(epetraFwdOpAdjointSupport) );
  const Epetra_RowMatrix &eRMOp  =
    Teuchos::dyn_cast<const Epetra_RowMatrix>(*epetraFwdOp);
  const Epetra_Map &map = eRMOp.OperatorDomainMap();
  RCP<Epetra_Vector>
    e_diag = Teuchos::rcp(new Epetra_Vector(map));
  eRMOp.ExtractDiagonalCopy(*e_diag);
  RCP< const VectorSpaceBase<double> >
    space = create_VectorSpace(Teuchos::rcp(new Epetra_Map(map)));
  RCP< const VectorBase<double> >
    diag = create_Vector(e_diag,space);
  Teuchos::set_extra_data<RCP<const LinearOpSourceBase<double> > >(
    fwdOpSrc, "Thyra::DiagonalEpetraLinearOpWithSolveFactory::fwdOpSrc",
    Teuchos::inOutArg(diag)
    );
  Teuchos::dyn_cast< DefaultDiagonalLinearOpWithSolve<double> >(*Op).initialize(
    Teuchos::rcp_implicit_cast<const VectorBase<double> >(diag)
    );
  // Above cast is questionable but should be okay based on use.
}


void DiagonalEpetraLinearOpWithSolveFactory::uninitializeOp(
  LinearOpWithSolveBase<double>                               *Op
  ,RCP<const LinearOpSourceBase<double> >    *fwdOpSrc
  ,RCP<const PreconditionerBase<double> >    *prec
  ,RCP<const LinearOpSourceBase<double> >    *approxFwdOpSrc
  ,ESupportSolveUse                                           *supportSolveUse
  ) const
{
  using Teuchos::get_extra_data;
  TEUCHOS_TEST_FOR_EXCEPT(Op==NULL);
  DefaultDiagonalLinearOpWithSolve<double>
    &diagOp = Teuchos::dyn_cast<DefaultDiagonalLinearOpWithSolve<double> >(*Op);
  RCP< const VectorBase<double> >
    diag = diagOp.getDiag();
  if( fwdOpSrc ) {
    if(diag.get()) {
      *fwdOpSrc =
        get_extra_data<RCP<const LinearOpSourceBase<double> > >(
          diag,"Thyra::DiagonalEpetraLinearOpWithSolveFactory::fwdOpSrc"
          );
    }
  }
  else {
    *fwdOpSrc = Teuchos::null;
  }
  if(prec) *prec = Teuchos::null; // We never keep a preconditioner!
  if(approxFwdOpSrc) *approxFwdOpSrc = Teuchos::null; // We never keep a preconditioner!
}


// Overridden from ParameterListAcceptor


void DiagonalEpetraLinearOpWithSolveFactory::setParameterList(
  RCP<Teuchos::ParameterList> const& paramList
  )
{}


RCP<Teuchos::ParameterList>
DiagonalEpetraLinearOpWithSolveFactory::getNonconstParameterList()
{
  return Teuchos::null;
}


RCP<Teuchos::ParameterList>
DiagonalEpetraLinearOpWithSolveFactory::unsetParameterList()
{
  return Teuchos::null;
}


RCP<const Teuchos::ParameterList>
DiagonalEpetraLinearOpWithSolveFactory::getParameterList() const
{
  return Teuchos::null;
}


RCP<const Teuchos::ParameterList>
DiagonalEpetraLinearOpWithSolveFactory::getValidParameters() const
{
  return Teuchos::null;
}


} // namespace Thyra
