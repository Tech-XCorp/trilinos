/*
// @HEADER
// ***********************************************************************
//
//          Tpetra: Templated Linear Algebra Services Package
//                 Copyright (2008) Sandia Corporation
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
// @HEADER
*/

#include "Kokkos_Performance_impl.hpp"
#include <iostream>

// Some dummy values for demo
struct resultstruct {
  double time1;
  double time2;
  int niters;
  double residual;
};

// example set up of a test entry
YAML::Node test_entry(const std::string testname,
  resultstruct results, const std::string& filename,
  int mpi_ranks, int teams, int threads, double tol_small, double tol_large) {

  // Create a configuration - all of the members are arbitrary and optional
  YAML::Node configuration;
  configuration["MPI_Ranks"] = mpi_ranks;
  configuration["Teams"] = teams;
  configuration["Threads"] = threads;
  configuration["Filename"] = filename;

  // Add test times - Keep Time keyword in name so that tolerance will work
  YAML::Node times;
  times["Time_1"] = Kokkos::ValueTolerance(results.time1,tol_large).as_string();
  times["Time_2"] = Kokkos::ValueTolerance(results.time2,tol_large).as_string();

  // Add test results - Keep Result in name so that tolerance will work
  times["Result_Iterations"] = Kokkos::ValueTolerance(results.niters,
    results.niters>0?results.niters-1:0, results.niters+1).as_string();
  times["Result_Residual"] =
    Kokkos::ValueTolerance(results.residual,tol_small).as_string();

  // Create the full test entry
  // TODO - This pattern is currently required for the performance code and it
  // would probably be better to encapsulate this internally so that the user
  // can't incorrectly set this up. However this matches the original Teuchos
  // formatting so will leave it like this for now until further discussion.
  YAML::Node entry; // the entry will have two bits added below
  entry[TestConfigurationString] = configuration;
  entry[TestResultsString] = times;

  // the test has a 'name' which is used for matching
  // if this changes a new entry will appear in the archive
  YAML::Node test;
  test[testname] = entry;
  return test;
}

int main(int argc, char *argv[]) {
  // This is the archive file that will store all test results
  std::string yaml_archive("Kokkos_YAMLPerformanceTestsExample.yaml");

  // this is the test name that will be archived
  // Changing this will create a new entry in the archive
  std::string testname = "ExampleTestName";

  // An optional hostname for test blocks - auto detected if left blank
  std::string hostname;

  // These are dummy values for a test entry
  // Changing these will create a new test entry in the archive
  std::string filename = "somefilename";
  const int mpi_ranks = 1;
  const int teams = 1;
  const int threads = 4;
  const double tol_small = 0.01;
  const double tol_large = 0.05;

  // These are dummy values for test results
  // Changing these triggers expected failure if prior results were saved
  resultstruct results;
  results.time1 = 10.0;
  results.time2 = 13.3;
  results.niters = 44;
  results.residual = 0.001;

  // Get the machine configuration from Kokkos
  // This can be further modified with test specific features if necessary
  YAML::Node machine_config = Kokkos::PerfTest_MachineConfig();

  // Create the test to pass to the archive
  YAML::Node test = test_entry( testname, results, filename,
    mpi_ranks, teams, threads, tol_small, tol_large);

  // Run the archiver which will either add the results, or compare them to
  // prior results if they already exist. This method will open the yaml,
  // import everything, do appropriate comparisons, then write out a new yaml.
  Kokkos::PerfTestResult resultCode = Kokkos::PerfTest_CheckOrAdd_Test
    (machine_config, test, yaml_archive, hostname);

  // Print results
  switch (resultCode) {
    case Kokkos::PerfTestPassed:
      std::cout << "End Result: TEST PASSED" << std::endl;
      break;
    case Kokkos::PerfTestFailed:
      std::cout << "End Result: FAILED" << std::endl;
      break;
    case Kokkos::PerfTestNewMachine:
      std::cout << "End Result: TEST PASSED. Adding new machine entry." << std::endl;
      break;
    case Kokkos::PerfTestNewConfiguration:
      std::cout << "End Result: TEST PASSED. Adding new machine configuration." << std::endl;
      break;
    case Kokkos::PerfTestNewTest:
      std::cout << "End Result: TEST PASSED. Adding new test entry." << std::endl;
      break;
    case Kokkos::PerfTestNewTestConfiguration:
      std::cout << "End Result: TEST PASSED. Adding new test entry configuration." << std::endl;
      break;
    case Kokkos::PerfTestUpdatedTest:
      std::cout << "End Result: TEST PASSED. Updating test entry." << std::endl;
      break;
    default:
      std::cout << "End Result: FAILED: Invalid comparison result." << std::endl;
  }

  return 0;
}
