/* ******************************************************************** */
/* See the file COPYRIGHT for a complete copyright notice, contact      */
/* person and disclaimer.                                               */
/* ******************************************************************** */
#include <stddef.h>
#include <sstream>
#include <string>

#include "Epetra_Map.h"
#include "Epetra_RowMatrix.h"
#include "MLAPI_Space.h"
#include "MLAPI_Workspace.h"
#include "ml_common.h"
#include "ml_config.h"
#include "ml_krylov.h"
#include "ml_operator.h"
#include "ml_rap.h"
#ifdef HAVE_ML_MLAPI
#include "Amesos_Lapack.h"
#include "Epetra_LinearProblem.h"
#include "Epetra_MultiVector.h"
#include "Epetra_Vector.h"
#include "MLAPI_Eig.h"
#include "MLAPI_Error.h"
#include "MLAPI_MultiVector.h"
#include "MLAPI_Operator.h"
#include "Teuchos_ParameterList.hpp"

namespace MLAPI {

// ======================================================================
double MaxEigAnorm(const Operator& Op, const bool DiagonalScaling)
{
  return(ML_Operator_MaxNorm(Op.GetML_Operator(), DiagonalScaling));
}

// ======================================================================
double MaxEigCG(const Operator& Op, const bool DiagonalScaling)
{

  ML_Krylov *kdata;
  double MaxEigen;

  kdata = ML_Krylov_Create(GetML_Comm());
  if (DiagonalScaling == false)
    kdata->ML_dont_scale_by_diag = ML_TRUE;
  else
    kdata->ML_dont_scale_by_diag = ML_FALSE;
  ML_Krylov_Set_PrintFreq(kdata, 0);
  ML_Krylov_Set_ComputeEigenvalues( kdata );
  ML_Krylov_Set_Amatrix(kdata, Op.GetML_Operator());
  ML_Krylov_Solve(kdata, Op.GetML_Operator()->outvec_leng, NULL, NULL);
  MaxEigen = ML_Krylov_Get_MaxEigenvalue(kdata);
  ML_Krylov_Destroy(&kdata);

  return(MaxEigen);
}

// ======================================================================
double MaxEigPowerMethod(const Operator& Op, const bool DiagonalScaling)
{

  ML_Krylov *kdata;
  double MaxEigen;

    kdata = ML_Krylov_Create(GetML_Comm());
    if (DiagonalScaling == false)
      kdata->ML_dont_scale_by_diag = ML_TRUE;
    else
      kdata->ML_dont_scale_by_diag = ML_FALSE;
    ML_Krylov_Set_PrintFreq(kdata, 0);
    ML_Krylov_Set_ComputeNonSymEigenvalues(kdata);
    ML_Krylov_Set_Amatrix(kdata, Op.GetML_Operator());
    ML_Krylov_Solve(kdata, Op.GetML_Operator()->outvec_leng, NULL, NULL);
    MaxEigen = ML_Krylov_Get_MaxEigenvalue(kdata);
  ML_Krylov_Destroy(&kdata);

  return(MaxEigen);
}

// ======================================================================
double MaxEigAnasazi(const Operator& Op, const bool DiagonalScaling)
{

  double MaxEigen = 0.0;

#if defined(HAVE_ML_EPETRA) && defined(HAVE_ML_ANASAxI) && defined(HAVE_ML_TEUCHOS)
  bool DiagScal;
  if (DiagonalScaling)
    DiagScal = ML_TRUE;
  else
    DiagScal = ML_FALSE;

  ML_Anasazi_Get_SpectralNorm_Anasazi(Op.GetML_Operator(), 0, 10, 1e-5,
                                      ML_FALSE, DiagScal, &MaxEigen);
#else
  //ML_THROW("Configure w/ --enable-epetra --enable-anasazi --enable-teuchos", -1);
  ML_THROW("Anasazi is no longer supported", -1);
#endif
  return(MaxEigen);
}

// ======================================================================
void Eig(const Operator& Op, MultiVector& ER, MultiVector& EI)
{
  int ierr;
  if (Op.GetDomainSpace() != Op.GetRangeSpace())
    ML_THROW("Matrix is not square", -1);

  ER.Reshape(Op.GetDomainSpace());
  EI.Reshape(Op.GetDomainSpace());

  Epetra_LinearProblem Problem;
  Problem.SetOperator(const_cast<Epetra_RowMatrix*>(Op.GetRowMatrix()));
  Amesos_Lapack Lapack(Problem);

  Epetra_Vector ER_Epetra(Op.GetRowMatrix()->RowMatrixRowMap());
  Epetra_Vector EI_Epetra(Op.GetRowMatrix()->RowMatrixRowMap());

  ierr = Lapack.GEEV(ER_Epetra, EI_Epetra);

  if (ierr)
    ML_THROW("GEEV returned error code = " + GetString(ierr), -1);

  for (int i = 0 ; i < ER.GetMyLength() ; ++i) {
    ER(i) = ER_Epetra[i];
    EI(i) = EI_Epetra[i];
  }
}

// ======================================================================
// FIXME: Add List
void Eigs(const Operator& A, int NumEigenvalues,
          MultiVector& ER, MultiVector& EI)
{

  if (A.GetDomainSpace() != A.GetRangeSpace())
    ML_THROW("Input Operator is not square", -1);

  int length = NumEigenvalues;
  double tol = 1e-3;
  int restarts = 1;
  int output = 10;
  bool PrintStatus = true;

  // 1.- set parameters for Anasazi
  Teuchos::ParameterList AnasaziList;
  // MatVec should be either "A" or "ML^{-1}A"
  AnasaziList.set("eigen-analysis: matrix operation", "A");
  AnasaziList.set("eigen-analysis: use diagonal scaling", false);
  AnasaziList.set("eigen-analysis: symmetric problem", false);
  AnasaziList.set("eigen-analysis: length", length);
  AnasaziList.set("eigen-analysis: block-size",1);
  AnasaziList.set("eigen-analysis: tolerance", tol);
  AnasaziList.set("eigen-analysis: restart", restarts);
  AnasaziList.set("eigen-analysis: output", output);
  AnasaziList.get("eigen-analysis: print current status",PrintStatus);

  // data to hold real and imag for eigenvalues and eigenvectors
  Space ESpace(-1, NumEigenvalues);
  ER.Reshape(ESpace);
  EI.Reshape(ESpace);

  // this is the starting value -- random
  Epetra_MultiVector EigenVectors(A.GetRowMatrix()->OperatorDomainMap(),
                                  NumEigenvalues);
  EigenVectors.Random();

  AnasaziList.set("eigen-analysis: action", "LM");

#ifdef HAVE_ML_ANASAxI
  ML_THROW("fixme...", -1);
  /* FIXME
  ML_Anasazi::Interface(A.GetRowMatrix(),EigenVectors,ER.GetValues(),
			EI.GetValues(), AnasaziList, 0, 0,
			&NumRealEigenvectors, &NumImagEigenvectors, 0);
                        */
#else
  ML_THROW("Anasazi is no longer supported", -1);
#endif

  return;
}

} // namespace MLAPI

#endif // HAVE_ML_MLAPI
