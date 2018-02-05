
TRIBITS_INCLUDE_DIRECTORIES(REQUIRED_DURING_INSTALLATION_TESTING ${PACKAGE_SOURCE_DIR}/test/src)

IF (WIN32 AND BUILD_SHARED_LIBS)
ELSE()

# Temp disable due to IWYU testing.
# Header is causing trouble - will try to get through the build
# and then come back and manually patch this up to see what’s going on.
#  TRIBITS_ADD_EXECUTABLE_AND_TEST(
#    FECrsMatrix_test
#    SOURCES cxx_main.cpp ExecuteTestProblems.cpp
#    TESTONLYLIBS epetratest
#    ARGS -v
#    COMM serial mpi
#    NUM_MPI_PROCS 1
#    FAIL_REGULAR_EXPRESSION "tests FAILED"
#    )

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    FEmatrix2
    SOURCES fematrix2.cpp
    TESTONLYLIBS epetratest
    ARGS -v
    COMM serial mpi
    NUM_MPI_PROCS 1
    FAIL_REGULAR_EXPRESSION "tests FAILED"
    )
ENDIF()
