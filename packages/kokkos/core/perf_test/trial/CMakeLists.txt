
TRIBITS_ADD_EXECUTABLE_AND_TEST(
  performance_tril
  SOURCES
    performance_trial.cpp
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  )
