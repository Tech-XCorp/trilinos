/* ******************************************************************** */
/* See the file COPYRIGHT for a complete copyright notice, contact      */
/* person and disclaimer.                                               */
/* ******************************************************************** */
#include <sstream>
#include <string>

#include "Epetra_DataAccess.h"
#include "Epetra_Map.h"
#include "MLAPI_Workspace.h"
#include "az_aztec.h"
#include "ml_common.h"
#include "ml_config.h"
#ifdef HAVE_ML_MLAPI
#include "Epetra_LinearProblem.h"
#include "Epetra_RowMatrix.h"
#include "Epetra_Vector.h"
#include "MLAPI_EpetraBaseOperator.h"
#include "MLAPI_Error.h"
#include "MLAPI_Krylov.h"
#include "MLAPI_MultiVector.h"
#include "MLAPI_Operator.h"
#ifdef HAVE_ML_AZTECOO
#include "AztecOO.h"
#endif
#include "Teuchos_ParameterList.hpp"

namespace MLAPI {

// ======================================================================
void Krylov(const Operator& A, const MultiVector& LHS,
            const MultiVector& RHS, const BaseOperator& Prec,
            Teuchos::ParameterList& List)
{
#ifndef HAVE_ML_AZTECOO
      std::cerr << "Please configure ML with --enable-aztecoo to use" << std::endl;
      std::cerr << "MLAPI Krylov solvers" << std::endl;
      exit(EXIT_FAILURE);
#else
  if (LHS.GetNumVectors() != 1)
    ML_THROW("FIXME: only one vector is currently supported", -1);

  Epetra_LinearProblem Problem;

  const Epetra_RowMatrix& A_Epetra = *(A.GetRowMatrix());

  Epetra_Vector LHS_Epetra(View,A_Epetra.OperatorDomainMap(),
                           (double*)&(LHS(0)));
  Epetra_Vector RHS_Epetra(View,A_Epetra.OperatorRangeMap(),
                           (double*)&(RHS(0)));

  // FIXME: this works only for Epetra-based operators
  Problem.SetOperator((const_cast<Epetra_RowMatrix*>(&A_Epetra)));
  Problem.SetLHS(&LHS_Epetra);
  Problem.SetRHS(&RHS_Epetra);

  AztecOO solver(Problem);

  EpetraBaseOperator Prec_Epetra(A_Epetra.OperatorDomainMap(),Prec);
  solver.SetPrecOperator(&Prec_Epetra);

  // get options from List
  int    NumIters = List.get("krylov: max iterations", 1550);
  double Tol      = List.get("krylov: tolerance", 1e-9);
  std::string type     = List.get("krylov: type", "gmres");
  int    output   = List.get("krylov: output level", GetPrintLevel());

  // set options in `solver'
  if (type == "cg")
    solver.SetAztecOption(AZ_solver, AZ_cg);
  else if (type == "cg_condnum")
    solver.SetAztecOption(AZ_solver, AZ_cg_condnum);
  else if (type == "gmres")
    solver.SetAztecOption(AZ_solver, AZ_gmres);
  else if (type == "gmres_condnum")
    solver.SetAztecOption(AZ_solver, AZ_gmres_condnum);
  else if (type == "fixed point")
    solver.SetAztecOption(AZ_solver, AZ_fixed_pt);
  else
    ML_THROW("krylov: type has incorrect value (" +
             type + ")", -1);

  solver.SetAztecOption(AZ_output, output);
  solver.Iterate(NumIters, Tol);
#endif

}

} // namespace MLAPI

#endif // HAVE_ML_MLAPI
