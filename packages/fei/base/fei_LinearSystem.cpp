/*--------------------------------------------------------------------*/
/*    Copyright 2005 Sandia Corporation.                              */
/*    Under the terms of Contract DE-AC04-94AL85000, there is a       */
/*    non-exclusive license for use of this work by or on behalf      */
/*    of the U.S. Government.  Export of this program may require     */
/*    a license from the United States Government.                    */
/*--------------------------------------------------------------------*/

#include <fei_LinearSystem.hpp>
#include <fei_MatrixGraph.hpp>
#include <snl_fei_LinearSystem_General.hpp>
#include <snl_fei_Utils.hpp>
#include <stddef.h>
#include <ostream>
#include <stdexcept>

#include "fei_DirichletBCManager.hpp"
#include "fei_Matrix.hpp"
#include "fei_SharedPtr.hpp"
#include "fei_Vector.hpp"
#include "fei_VectorSpace.hpp"
#include "fei_console_ostream.hpp"
#include "fei_iostream.hpp"

//----------------------------------------------------------------------------
fei::LinearSystem::LinearSystem(fei::SharedPtr<fei::MatrixGraph>& matrixGraph)
 : matrix_(),
   soln_(),
   rhs_(),
   matrixGraph_(matrixGraph),
   dbcManager_(NULL)
{
}

//----------------------------------------------------------------------------
fei::LinearSystem::~LinearSystem()
{
  delete dbcManager_;

  for(unsigned i=0; i<attributeNames_.size(); ++i) {
    delete [] attributeNames_[i];
  }
}

//----------------------------------------------------------------------------
fei::SharedPtr<fei::LinearSystem>
fei::LinearSystem::Factory::createLinearSystem(fei::SharedPtr<fei::MatrixGraph>& matrixGraph)
{
  fei::SharedPtr<fei::LinearSystem>
    linsys(new snl_fei::LinearSystem_General(matrixGraph));

  return(linsys);
}

//----------------------------------------------------------------------------
void fei::LinearSystem::setMatrix(fei::SharedPtr<fei::Matrix>& matrix)
{
  matrix_ = matrix;
}

//----------------------------------------------------------------------------
int fei::LinearSystem::putAttribute(const char* name,
                                    void* attribute)
{
  snl_fei::storeNamedAttribute(name, attribute,
                               attributeNames_, attributes_);
  return(0);
}

//----------------------------------------------------------------------------
int fei::LinearSystem::getAttribute(const char* name,
                                    void*& attribute)
{
  attribute = snl_fei::retrieveNamedAttribute(name, attributeNames_, attributes_);
  return(attribute==NULL ? -1 : 0);
}

//----------------------------------------------------------------------------
int fei::LinearSystem::loadEssentialBCs(int numIDs,
                                 const int* IDs,
                                 int idType,
                                 int fieldID,
                                 int offsetIntoField,
                                 const double* prescribedValues)
{
  if (dbcManager_ == NULL) {
    dbcManager_ = new fei::DirichletBCManager(matrixGraph_->getRowSpace());
  }

  try {
    dbcManager_->addBCRecords(numIDs, idType, fieldID, offsetIntoField,
                              IDs, prescribedValues);
  }
  catch(std::runtime_error& exc) {
    fei::console_out() << exc.what()<<FEI_ENDL;
    return(-1);
  }

  return(0);
}

//----------------------------------------------------------------------------
int fei::LinearSystem::loadEssentialBCs(int numIDs,
                                 const int* IDs,
                                 int idType,
                                 int fieldID,
                                 const int* offsetsIntoField,
                                 const double* prescribedValues)
{
  if (dbcManager_ == NULL) {
    dbcManager_ = new fei::DirichletBCManager(matrixGraph_->getRowSpace());
  }

  try {
    dbcManager_->addBCRecords(numIDs, idType, fieldID, IDs, offsetsIntoField,
                              prescribedValues);
  }
  catch(std::runtime_error& exc) {
    fei::console_out() << exc.what()<<FEI_ENDL;
    return(-1);
  }

  return(0);
}

