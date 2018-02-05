#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>
#include <stdlib.h>
#include <algorithm>
#include <sstream>
#include <vector>

#include "RTOpPack_Types.hpp"
#include "Teuchos_ArrayRCP.hpp"
#include "Teuchos_ArrayView.hpp"
#include "Teuchos_Comm.hpp"
#include "Teuchos_Describable.hpp"
#include "Teuchos_FancyOStream.hpp"
#include "Teuchos_LocalTestingHelpers.hpp"
#include "Teuchos_Ptr.hpp"
#include "Teuchos_RCPDecl.hpp"
#include "Teuchos_TestingHelpers.hpp"
#include "Teuchos_UnitTestHelpers.hpp"
#include "Teuchos_VerbosityLevel.hpp"
#include "Thyra_DefaultProductVectorSpace_decl.hpp"
#include "Thyra_DefaultSpmdVectorSpace_decl.hpp"
#include "Thyra_LinearOpBase_decl.hpp"
#include "Thyra_MultiVectorBase_decl.hpp"
#include "Thyra_MultiVectorStdOps_decl.hpp"
#include "Thyra_OperatorVectorTypes.hpp"
#include "Thyra_ProductMultiVectorBase.hpp"
#include "Thyra_VectorSpaceBase_decl.hpp"

TEUCHOS_UNIT_TEST(thyra_vec, apply_test)
{
  typedef Thyra::Ordinal Ordinal;

  Teuchos::RCP<const Teuchos::Comm<Ordinal> > comm = Teuchos::DefaultComm<Ordinal>::getComm();

  int localSz = 6/comm->getSize();

  // 
  // this is the corrector answer
  //  x = [ [2,2,2,2,2,2], [2] ]
  //  y = [ [3,3,3,3,3,3], [-7] ]
  //
  //  x'*y = 6*6 + 2*(-7) = 36-14 = 22
  //

  //
  // for two processors (currently, and this is wrong!!!): 
  //       x' * y = 6*6 + 2*(-7) + 2*(-7) = 36-28 = 8
  // for three processors (currently, and this is wrong!!!): 
  //       x' * y = 6*6 + 2*(-7) + 2*(-7) + 2*(-7) = 36-42 = -6
  //
  
  // build the components of the product vector space
  Teuchos::RCP<const Thyra::VectorSpaceBase<double> > vs_0 = Thyra::defaultSpmdVectorSpace<double>(comm,localSz,-1);
  Teuchos::RCP<const Thyra::VectorSpaceBase<double> > vs_1 = Thyra::locallyReplicatedDefaultSpmdVectorSpace<double>(comm,1);
  std::vector<Teuchos::RCP<const Thyra::VectorSpaceBase<double> >  > array;
  array.push_back(vs_0);
  array.push_back(vs_1);
  Teuchos::RCP<const Thyra::VectorSpaceBase<double> > vs = Thyra::productVectorSpace<double>(array);

  Teuchos::RCP<Thyra::MultiVectorBase<double> > x = Thyra::createMembers(*vs,1);
  Teuchos::RCP<Thyra::MultiVectorBase<double> > y = Thyra::createMembers(*vs,1);

  Thyra::assign(x.ptr(),2.0);
  Thyra::assign(Teuchos::rcp_dynamic_cast<Thyra::ProductMultiVectorBase<double> >(y)->getNonconstMultiVectorBlock(0).ptr(),3.0);
  Thyra::assign(Teuchos::rcp_dynamic_cast<Thyra::ProductMultiVectorBase<double> >(y)->getNonconstMultiVectorBlock(1).ptr(),-7.0);

  Teuchos::SerialDenseMatrix<int,double> C(1,1);
  out << "X = \n" << Teuchos::describe(*x,Teuchos::VERB_EXTREME) << std::endl;
  out << "Y = \n" << Teuchos::describe(*y,Teuchos::VERB_EXTREME) << std::endl;

  int m = x->domain()->dim();
  int n = y->domain()->dim();

  // Create a view of the B object!
  Teuchos::RCP< Thyra::MultiVectorBase<double>  > C_thyra = Thyra::createMembersView(x->domain(),
          RTOpPack::SubMultiVectorView<double>( 0, m, 0, n,Teuchos::arcpFromArrayView(Teuchos::arrayView(&C(0,0), C.stride()*C.numCols())),
          C.stride()));

  Thyra::apply<double>(*x, Thyra::CONJTRANS, *y, C_thyra.ptr(), 1.0);

  out << "C = \n";
  C.print(out);

  TEST_EQUALITY(C(0,0),22.0)
}

TEUCHOS_UNIT_TEST(thyra_vec, apply_test2)
{
  typedef Thyra::Ordinal Ordinal;

  Teuchos::RCP<const Teuchos::Comm<Ordinal> > comm = Teuchos::DefaultComm<Ordinal>::getComm();

  // 
  // this is the corrector answer
  //  x = [ 2 ]
  //  y = [ -7 ]
  //
  //  x'*y = 2*(-7) = -14
  //

  //
  // for two processors (currently, and this is wrong!!!): 
  //       x' * y = 2*(-7) + 2*(-7) = -28
  // for three processors (currently, and this is wrong!!!): 
  //       x' * y = 2*(-7) + 2*(-7) + 2*(-7) = -42
  //
  
  // build the components of the product vector space
  Teuchos::RCP<const Thyra::VectorSpaceBase<double> > vs = Thyra::locallyReplicatedDefaultSpmdVectorSpace<double>(comm,1);

  Teuchos::RCP<Thyra::MultiVectorBase<double> > x = Thyra::createMembers(*vs,1);
  Teuchos::RCP<Thyra::MultiVectorBase<double> > y = Thyra::createMembers(*vs,1);

  Thyra::assign(x.ptr(),2.0);
  Thyra::assign(y.ptr(),-7.0);

  Teuchos::SerialDenseMatrix<int,double> C(1,1);
  // out << "X = \n" << Teuchos::describe(*x,Teuchos::VERB_EXTREME) << std::endl;
  // out << "Y = \n" << Teuchos::describe(*y,Teuchos::VERB_EXTREME) << std::endl;

  int m = x->domain()->dim();
  int n = y->domain()->dim();

  // Create a view of the B object!
  Teuchos::RCP< Thyra::MultiVectorBase<double>  > C_thyra = Thyra::createMembersView(x->domain(),
          RTOpPack::SubMultiVectorView<double>( 0, m, 0, n,Teuchos::arcpFromArrayView(Teuchos::arrayView(&C(0,0), C.stride()*C.numCols())),
          C.stride()));

  out << "RUNNING APPLY" << std::endl;
  Thyra::apply<double>(*x, Thyra::CONJTRANS, *y, C_thyra.ptr(), 1.0);
  out << "END APPLY" << std::endl;

  // out << "C = \n";
  // C.print(out);

  TEST_EQUALITY(C(0,0),-14.0)
}
