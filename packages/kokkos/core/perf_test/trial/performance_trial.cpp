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

#include <Kokkos_YAMLPerfTestArchive.hpp>
#include <iostream>

int main(int argc, char *argv[]) {

  int numgpus = 1;
  int skipgpu = 999;
  std::string filename;
  std::string filename_vector;
  std::string testarchive("Kokkos_PerformanceTests.xml");
  std::string hostname;

  int myRank = 0;
#ifdef HAVE_MPI
  (void) MPI_Comm_rank (MPI_COMM_WORLD, &myRank);
#endif // HAVE_MPI

  int device = myRank % numgpus;
  if(device>=skipgpu) device++;

  using std::cout;
  using std::endl;
  
  std::cout << "running..." << std::endl;
  
  // Print results.
    Kokkos::YAMLTestNode machine_config = Kokkos::PerfTest_MachineConfig();
    Kokkos::YAMLTestNode test;
    Kokkos::PerfTestResult comparison_result =
      Kokkos::PerfTest_CheckOrAdd_Test (machine_config, test, testarchive, hostname);
    switch (comparison_result) {
      case Kokkos::PerfTestPassed:
        cout << "PASSED" << endl;
        break;
      case Kokkos::PerfTestFailed:
        cout << "FAILED" << endl;
        break;
      case Kokkos::PerfTestNewMachine:
        cout << "PASSED. Adding new machine entry." << endl;
        break;
      case Kokkos::PerfTestNewConfiguration:
        cout << "PASSED. Adding new machine configuration." << endl;
        break;
      case Kokkos::PerfTestNewTest:
        cout << "PASSED. Adding new test entry." << endl;
        break;
      case Kokkos::PerfTestNewTestConfiguration:
        cout << "PASSED. Adding new test entry configuration." << endl;
        break;
      case Kokkos::PerfTestUpdatedTest:
        cout << "PASSED. Updating test entry." << endl;
        break;
    default:
      cout << "FAILED: Invalid comparison result." << endl;
    }

  return EXIT_SUCCESS;
  
  return 0;
}
