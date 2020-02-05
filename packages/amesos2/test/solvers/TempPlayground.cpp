#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_TestingHelpers.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_ParameterXMLFileReader.hpp>
#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_Comm.hpp>

#include <Tpetra_Core.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_CrsMatrix.hpp>

#    include <cuda.h>
#    include <cusolverSp.h>
#    include <cusolverDn.h>
#    include <cusparse.h>
#    include <cuComplex.h>

int main(int argc, char*argv[])
{
  Tpetra::ScopeGuard tpetraScope(&argc,&argv);

#ifdef HAVE_AMESOS2_CUSOLVER
    //initialize our test cases
    const int size = 3; // this->globalNumRows_
    const int nnz = 6; // this->globalNumNonZeros_
    int sing = 0;

    //float values[] = {0,0,0,0} ;
    float values[nnz] = {1,2,3,4,5,6}; // host_nzvals_view_
    int colIdx[nnz] = {0,0,1,0,1,2}; // host_cols_view_
    int rowPtr[size+1] = {0, 1,3,6}; // host_row_ptr_view_

    float b[size] = {4,-6,7};
    float x[size]= {0,0,0} ;
    cusolverStatus_t cso;
    cusolverSpHandle_t solver_handle ;
    cso = cusolverSpCreate(&solver_handle) ;
    assert(cso == CUSOLVER_STATUS_SUCCESS);
    cusparseStatus_t csp;
    cusparseMatDescr_t descr = 0;

    csp = cusparseCreateMatDescr(&descr);
    assert(csp == CUSPARSE_STATUS_SUCCESS);
    csp = cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    assert(csp == CUSPARSE_STATUS_SUCCESS);
    csp = cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
    assert(csp == CUSPARSE_STATUS_SUCCESS);
    cso = cusolverSpScsrlsvluHost(solver_handle, size, nnz, descr, values, rowPtr, colIdx, b, 0.0,0, x, &sing);
    assert(cso == CUSOLVER_STATUS_SUCCESS);
    printf("%f\n",x[0]);
    printf("%f\n",x[1]);
    printf("%f\n",x[2]);
#endif

  std::cout << "\nEnd Result: TEST PASSED" << std::endl;

  return 0;
}
