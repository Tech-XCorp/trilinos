/*
//@HEADER
// ************************************************************************
//
//               KokkosKernels 0.9: Linear Algebra and Graph Kernels
//                 Copyright 2017 Sandia Corporation
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
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/
#include <Kokkos_Blas1.hpp>
#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_Comm.hpp>
#include "Kokkos_Performance_impl.hpp"
#ifdef HAVE_MPI
#  include <Teuchos_DefaultMpiComm.hpp>
#else
#  include <Teuchos_DefaultSerialComm.hpp>
#endif // HAVE_MPI

using Teuchos::Comm;
using Teuchos::CommandLineProcessor;
using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::Time;
using Teuchos::TimeMonitor;

// Create a new timer with the given name if it hasn't already been
// created, else get the previously created timer with that name.
RCP<Time> getTimer (const std::string& timerName) {
  RCP<Time> timer = TimeMonitor::lookupCounter (timerName);
  if (timer.is_null ()) {
    timer = TimeMonitor::getNewCounter (timerName);
  }
  return timer;
}

bool
benchmarkKokkos (std::ostream& out,
                 const int lclNumRows,
                 const int numTrials)
{
  using std::endl;
  typedef Kokkos::View<double*, Kokkos::LayoutLeft> vector_type;

  RCP<Time> vecCreateTimer = getTimer ("Kokkos: Vector: Create");
  RCP<Time> vecFillTimer = getTimer ("Kokkos: Vector: Fill");
  RCP<Time> vecDotTimer = getTimer ("Kokkos: Vector: Dot");

  // Benchmark creation of a Vector.
  vector_type x;
  {
    TimeMonitor timeMon (*vecCreateTimer);
    // This benchmarks both vector creation and vector destruction.
    for (int k = 0; k < numTrials; ++k) {
      x = vector_type ("x", lclNumRows);
    }
  }

  // Benchmark filling a Vector.
  {
    TimeMonitor timeMon (*vecFillTimer);
    for (int k = 0; k < numTrials; ++k) {
      Kokkos::deep_copy (x, 1.0);
    }
  }

  vector_type y ("y", lclNumRows);
  Kokkos::deep_copy (y, -1.0);

  // Benchmark computing the dot product of two Vectors.
  double dotResults[2];
  dotResults[0] = 0.0;
  dotResults[1] = 0.0;
  {
    TimeMonitor timeMon (*vecDotTimer);
    for (int k = 0; k < numTrials; ++k) {
      // "Confuse" the compiler so it doesn't optimize away the dot() calls.
      dotResults[k % 2] = KokkosBlas::dot (x, y);
    }
  }

  if (numTrials > 0) {
    const double expectedResult = static_cast<double> (lclNumRows) * -1.0;
    if (dotResults[0] != expectedResult) {
      out << "Kokkos dot product result is wrong!  Expected " << expectedResult
          << " but got " << dotResults[0] << " instead." << endl;
      return false;
    } else {
      return true;
    }
  }
  return true;
}


bool
benchmarkRaw (std::ostream& out,
              const int lclNumRows,
              const int numTrials)
{
  using std::endl;
  RCP<Time> vecCreateTimer = getTimer ("Raw: Vector: Create");
  RCP<Time> vecFillTimer = getTimer ("Raw: Vector: Fill");
  RCP<Time> vecDotTimer = getTimer ("Raw: Vector: Dot");

  // Benchmark creation of a Vector.
  double* x = 0 ;
  {
    TimeMonitor timeMon (*vecCreateTimer);
    // This benchmarks both vector creation and vector destruction.
    for (int k = 0; k < numTrials; ++k) {
      x = new double [lclNumRows];
      memset (x, 0, lclNumRows * sizeof (double));
      if (k + 1 < numTrials) {
        delete [] x;
      }
    }
  }

  // Benchmark filling a Vector.
  {
    TimeMonitor timeMon (*vecFillTimer);
    for (int k = 0; k < numTrials; ++k) {
      for (int i = 0; i < lclNumRows; ++i) {
        x[i] = 1.0;
      }
    }
  }

  double* y = new double [lclNumRows];
  for (int i = 0; i < lclNumRows; ++i) {
    y[i] = -1.0;
  }

  // Benchmark computing the dot product of two Vectors.
  double dotResults[2];
  dotResults[0] = 0.0;
  dotResults[1] = 0.0;
  {
    TimeMonitor timeMon (*vecDotTimer);
    for (int k = 0; k < numTrials; ++k) {
      double sum = 0.0;
      for (int i = 0; i < lclNumRows; ++i) {
        sum += x[i] * y[i];
      }
      // "Confuse" the compiler so it doesn't optimize away the loops.
      dotResults[k % 2] = sum;
    }
  }

  if (x != NULL) {
    delete [] x;
    x = NULL;
  }
  if (y != NULL) {
    delete [] y;
    y = NULL;
  }

  if (numTrials == 0) {
    return true; // trivially
  }
  else { // numTrials > 0
    const double expectedResult = static_cast<double> (lclNumRows) * -1.0;
    if (dotResults[0] != expectedResult) {
      out << "Raw dot product result is wrong!  Expected " << expectedResult
          << " but got " << dotResults[0] << " instead." << endl;
      return false;
    } else {
      return true;
    }
  }
}

void callPerformanceArchiver(double tolerance, int lclNumRows, int numTrials) {
  // This is the archive file that will store all test results
  std::string yaml_archive("performance_demo_blas1.yaml");

  // this is the test name that will be archived
  // Changing this will create a new entry in the archive
  std::string testname = "blas1";

  // An optional hostname for test blocks - auto detected if left blank
  std::string hostname;

  // Get the machine configuration from Kokkos
  // This can be further modified with test specific features if necessary
  YAML::Node machine_config = Kokkos::PerfTest_MachineConfig();

  // Create a configuration - all of the members are arbitrary and optional
  YAML::Node configuration;
  configuration["lclNumRows"] = lclNumRows;
  configuration["numTrials"] = numTrials;

  // Add test times - Keep Time keyword in name so that tolerance will work
  double kokkosVectorCreateTime = TimeMonitor::lookupCounter("Kokkos: Vector: Create")->totalElapsedTime();
  double kokkosVectorDotTime = TimeMonitor::lookupCounter("Kokkos: Vector: Dot")->totalElapsedTime();
  double kokkosVectorFill = TimeMonitor::lookupCounter("Kokkos: Vector: Fill")->totalElapsedTime();
  double rawVectorCreateTime = TimeMonitor::lookupCounter("Raw: Vector: Create")->totalElapsedTime();
  double rawVectorDotTime = TimeMonitor::lookupCounter("Raw: Vector: Dot")->totalElapsedTime();
  double rawVectorFill = TimeMonitor::lookupCounter("Raw: Vector: Fill")->totalElapsedTime();


  YAML::Node times;
  
  // For now use the same names for the YAML archive
  // However note we must use the Time keywork for tolerances to work
  // Otherwise the archiver will be looking for an exact match
  times["Time Kokkos: Vector: Create"] = Kokkos::ValueTolerance(kokkosVectorDotTime,tolerance).as_string();
  times["Time Kokkos: Vector: Dot"] = Kokkos::ValueTolerance(kokkosVectorDotTime,tolerance).as_string();
  times["Time eKokkos: Vector: Fill"] = Kokkos::ValueTolerance(kokkosVectorDotTime,tolerance).as_string();
  times["Time Raw: Vector: Create"] = Kokkos::ValueTolerance(kokkosVectorDotTime,tolerance).as_string();
  times["Time Raw: Vector: Dot"] = Kokkos::ValueTolerance(kokkosVectorDotTime,tolerance).as_string();
  times["Time Raw: Vector: Fill"] = Kokkos::ValueTolerance(kokkosVectorDotTime,tolerance).as_string();

  // Add test results - Keep Result in name so that tolerance will work
  /* TODO - Not currently used
  times["Result_Iterations"] = Kokkos::ValueTolerance(results.niters,
    results.niters>0?results.niters-1:0, results.niters+1).as_string();
  times["Result_Residual"] =
    Kokkos::ValueTolerance(results.residual,tol_small).as_string();
  */

  
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
}

int
main (int argc, char* argv[])
{
  using std::cout;
  using std::endl;
  Teuchos::oblackholestream blackHole;
  Teuchos::GlobalMPISession mpiSession (&argc, &argv, &blackHole);
  Kokkos::initialize (argc, argv);

#ifdef HAVE_MPI
  RCP<const Comm<int> > comm = rcp (new Teuchos::MpiComm<int> (MPI_COMM_WORLD));
#else
  RCP<const Comm<int> > comm = rcp (new Teuchos::SerialComm<int> ());
#endif // HAVE_MPI

  //const int numProcs = comm->getSize (); // unused
  const int myRank = comm->getRank ();

  // Benchmark parameters
  int lclNumRows = 10000;
  int numTrials = 1000;

  bool runKokkos = true;
  bool runRaw = true;

  CommandLineProcessor cmdp;
  cmdp.setOption ("lclNumRows", &lclNumRows, "Number of global indices "
                  "owned by each process");
  cmdp.setOption ("numTrials", &numTrials, "Number of timing loop iterations for each event to time");
  cmdp.setOption ("runKokkos", "noKokkos", &runKokkos,
                  "Whether to run the Kokkos benchmark");
  cmdp.setOption ("runRaw", "noRaw", &runRaw,
                  "Whether to run the raw benchmark");
  const CommandLineProcessor::EParseCommandLineReturn parseResult =
    cmdp.parse (argc, argv);
  if (parseResult == CommandLineProcessor::PARSE_HELP_PRINTED) {
    // The user specified --help at the command line to print help
    // with command-line arguments.  We printed help already, so quit
    // with a happy return code.
    return EXIT_SUCCESS;
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(
      parseResult != CommandLineProcessor::PARSE_SUCCESSFUL,
      std::invalid_argument, "Failed to parse command-line arguments.");
    TEUCHOS_TEST_FOR_EXCEPTION(
      lclNumRows < 0, std::invalid_argument,
      "lclNumRows must be nonnegative.");
  }

  if (myRank == 0) {
    cout << endl << "---" << endl
         << "Command-line options:" << endl
         << "  lclNumRows: " << lclNumRows << endl
         << "  numTrials: " << numTrials << endl
         << "  runKokkos: " << (runKokkos ? "true" : "false") << endl
         << "  runRaw: " << (runRaw ? "true" : "false") << endl
         << endl;
  }

  // Run the benchmark
  bool success = true;
  if (runKokkos) {
    const bool lclSuccess = benchmarkKokkos (cout, lclNumRows, numTrials);
    success = success && lclSuccess;
  }
  if (runRaw) {
    const bool lclSuccess = benchmarkRaw (cout, lclNumRows, numTrials);
    success = success && lclSuccess;
  }

  TimeMonitor::report (comm.ptr (), cout);
  
  const bool bUsePerformanceArchiver = true;
  
  if(bUsePerformanceArchiver) {
    double tolerance = 0.1;
    callPerformanceArchiver(tolerance, lclNumRows, numTrials);
  }
  else {
    if (success) {
      cout << "End Result: TEST PASSED" << endl;
    } else {
      cout << "End Result: TEST FAILED" << endl;
    }
  }

  Kokkos::finalize ();
  return EXIT_SUCCESS;
}
