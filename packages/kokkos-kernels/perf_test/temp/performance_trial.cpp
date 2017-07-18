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
YAML::Node test_entry(resultstruct results, const std::string& filename,
  int mpi_ranks, int teams, int threads, double tol_small, double tol_large) {

  // Create a configuration
  YAML::Node configuration;
  configuration["MPI_Ranks"] = mpi_ranks;
  configuration["Teams"] = teams;
  configuration["Threads"] = threads;
  configuration["Filename"] = filename;

  // Create a times block - Keep Time in name - part of key work searching
  YAML::Node times;
  times["Time_1"] = Kokkos::ValueTolerance(results.time1,tol_large).as_string();
  times["Time_2"] = Kokkos::ValueTolerance(results.time2,tol_large).as_string();
  // Keep Result in name - part of key word searching
  times["Result_Iterations"] = Kokkos::ValueTolerance(results.niters,
    results.niters>0?results.niters-1:0, results.niters+1).as_string();
  times["Result_Residual"] =
    Kokkos::ValueTolerance(results.residual,tol_small).as_string();

  // Create the full test entry
  YAML::Node entry;
  entry["TestConfiguration"] = configuration;
  entry["TestResults"] = times;
  
  YAML::Node test;
  test["ExampleTestName"] = entry;
  return test;
}

int main(int argc, char *argv[]) {

  // define some values for a test mock up
  std::string yaml_archive("Kokkos_YAMLPerformanceTestsExample.yaml");
  std::string filename = "somefilename";
  std::string hostname;

  const int mpi_ranks = 1;
  const int teams = 1;
  const int threads = 4;
  const double tol_small = 0.01;
  const double tol_large = 0.05;

  resultstruct results;
  results.time1 = 10.0;
  results.time2 = 13.3;
  results.niters = 44;
  results.residual = 0.001;

  // Process the test
  YAML::Node machine_config = Kokkos::PerfTest_MachineConfig();
  YAML::Node test = test_entry( results, filename, mpi_ranks, teams, threads,
    tol_small, tol_large);
  Kokkos::PerfTestResult resultCode = Kokkos::PerfTest_CheckOrAdd_Test
    (machine_config, test, yaml_archive, hostname);
  
  std::cout << Kokkos::message_from_test_result(resultCode) << std::endl;
  
  return 0;
}
