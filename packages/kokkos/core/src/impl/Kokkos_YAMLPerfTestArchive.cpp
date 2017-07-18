// @HEADER
// ***********************************************************************
//
//                    Teuchos: Common Tools Package
//                 Copyright (2004) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
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
// ***********************************************************************
// @HEADER

#include <iostream>
#include <fstream>
#include <Kokkos_YAMLPerfTestArchive.hpp>

// For determining hostname
#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#include <Winsock2.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <unistd.h>
#endif

namespace Kokkos {

ValueTolerance::ValueTolerance() {
  value = 0;
  lower = 0;
  upper = 0;
  tolerance = 0;
  use_tolerance = true;
}

ValueTolerance::ValueTolerance(double val, double tol) {
  value = val;
  lower = 0;
  upper = 0;
  tolerance = tol;
  use_tolerance = true;
}

ValueTolerance::ValueTolerance(double val, double low, double up) {
  value = val;
  upper = up;
  lower = low;
  tolerance = 0;
  use_tolerance = false;
}

ValueTolerance::ValueTolerance(std::string str) {
  from_string(str);
}

bool ValueTolerance::operator ==(ValueTolerance& rhs) {
  return (value == rhs.value) &&
         (tolerance == rhs.tolerance) &&
         (lower == rhs.lower) &&
         (upper == rhs.upper) &&
         (use_tolerance == rhs.use_tolerance);
}

std::string ValueTolerance::as_string(){
  std::ostringstream strs;
  if(use_tolerance)
    strs << value << " , " << tolerance;
  else
    strs << value << " , " << lower << " , " << upper;
  return  strs.str();
}

void ValueTolerance::from_string(const std::string& valtol_str) {
  std::string value_str = valtol_str.substr(0,valtol_str.find(","));
  value = atof(value_str.c_str());
  std::string tol_str = valtol_str.substr(valtol_str.find(",")+1);
  if(tol_str.find(",")<=tol_str.length()) {
    use_tolerance = false;
    std::string lower_str = tol_str.substr(0,tol_str.find(","));
    lower = atof(lower_str.c_str());
    std::string upper_str = tol_str.substr(tol_str.find(",")+1);
    upper = atof(upper_str.c_str());
  } else {
    use_tolerance = true;
    tolerance = atof(tol_str.c_str());
  }
}

YAML::Node PerfTest_MachineConfig() {

  // Get CPUName, Number of Sockets, Number of Cores, Number of Hyperthreads
  std::string cpuname("Undefined");
  unsigned int threads = 0;
  unsigned int cores_per_socket = 0;
  unsigned int highest_socketid = 0;

  {
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    if((cpuinfo.rdstate()&cpuinfo.failbit)) std::cout<<"Failed to open filen\n";
    while (!cpuinfo.eof() && !(cpuinfo.rdstate()&cpuinfo.failbit)) {
      getline (cpuinfo,line);
      if (line.find("model name") < line.size()) {
        cpuname = line.substr(line.find(":")+2);
        threads++;
      }
      if (line.find("physical id") < line.size()) {
        unsigned int socketid = atoi(line.substr(line.find(":")+2).c_str());
        highest_socketid = highest_socketid>socketid?highest_socketid:socketid;
      }
      if (line.find("cpu cores") < line.size()) {
        cores_per_socket = atoi(line.substr(line.find(":")+2).c_str());
      }
    }
  }

  YAML::Node machine_config;
  machine_config["Compiler"] = KOKKOS_COMPILER_NAME;
  machine_config["Compiler_Version"] = KOKKOS_COMPILER_VERSION;
  machine_config["CPU_Name"] = cpuname;
  machine_config["CPU_Sockets"] = highest_socketid+1;
  machine_config["CPU_Cores_Per_Socket"] = cores_per_socket;
  machine_config["CPU_Total_HyperThreads"] = threads;

  return machine_config;
}

bool hasSameElements(YAML::Node a, YAML::Node b, int rec = 0) {
  if(a.size()!=b.size()) {
    return false;
  }

  for (YAML::const_iterator i = a.begin(); i != a.end(); ++i) {
    std::string cat_name = i->first.Scalar();
    // validate we can find this cat in b
    if(!b[cat_name]) {
      return false;
    }
    YAML::Node sub_a = i->second;
    YAML::Node sub_b = b[cat_name];
    if(sub_a.Scalar() != sub_b.Scalar()) {
      return false;
    }

    if(!hasSameElements(sub_a, sub_b)) {
      return false;
    }
  }
  
  return true;
}

std::string message_from_test_result(PerfTestResult result) {
  // Print results.
  switch (result) {
    case Kokkos::PerfTestPassed:
      return "PASSED";
      break;
    case Kokkos::PerfTestFailed:
      return "FAILED";
      break;
    case Kokkos::PerfTestNewMachine:
      return "PASSED. Adding new machine entry.";
      break;
    case Kokkos::PerfTestNewConfiguration:
      return "PASSED. Adding new machine configuration.";
      break;
    case Kokkos::PerfTestNewTest:
      return "PASSED. Adding new test entry.";
      break;
    case Kokkos::PerfTestNewTestConfiguration:
      return "PASSED. Adding new test entry configuration.";
      break;
    case Kokkos::PerfTestUpdatedTest:
      return "PASSED. Updating test entry.";
      break;
    default:
      return "FAILED: Invalid comparison result.";
  }
}

PerfTestResult
PerfTest_CheckOrAdd_Test (YAML::Node machine_config,
                          YAML::Node new_test_with_name,
                          const std::string filename,
                          const std::string ext_hostname)
{
  YAML::Node database;

  PerfTestResult return_value = PerfTestPassed;
  bool is_new_config = true;

  // Open YAML File whhich stores test database
  if (std::ifstream (filename.c_str ())) {
    database = YAML::LoadFile(filename);
  }

  // Get Current Hostname
  char hostname[256];
  memset (hostname, 0, 256);
  if (ext_hostname.empty ()) {
    gethostname (hostname, 255);
  } else {
    strncat (hostname, ext_hostname.c_str (), 255);
  }

  // this gets the test info without the test name
  YAML::Node new_test_entry = new_test_with_name.begin()->second;

  // get the actual test name which will be used for matching in the database
  std::string new_test_entry_name = new_test_with_name.begin()->first.Scalar();

  // make sure the test is set up properly
  if(!new_test_entry["TestConfiguration"]) {
    throw std::logic_error("A TestEntry needs to have a child \"TestConfiguration\".");
  }
  if(!new_test_entry["TestResults"]) {
    throw std::logic_error("A TestEntry needs to have \"TestResults\".");
  }
            
  // Does hostname exist?
  if (database[hostname]) {
    YAML::Node machine = database[hostname];

    // Find matching machine configuration
    for (size_t machine_index = 0; machine_index < machine.size(); ++machine_index) {
      YAML::Node configuration = machine[machine_index];
      if(!configuration["MachineConfiguration"] || !configuration["Tests"]) {
        throw std::logic_error("Configuration must has child MachineConfiguration and a child \"Tests\".");
      }

      YAML::Node machine_configuration = configuration["MachineConfiguration"];
      YAML::Node old_tests = configuration["Tests"];
      if (hasSameElements(machine_configuration, machine_config)) {
        is_new_config = false;

        // Find existing test with same tag as the new test
        if(old_tests[new_test_entry_name]) {
          YAML::Node old_test_array = old_tests[new_test_entry_name];
          int match_test_index = -1;
          for (size_t entry_index = 0; entry_index < old_test_array.size(); ++entry_index) {
            YAML::Node old_test_entry = old_test_array[entry_index];
            if (hasSameElements(old_test_entry["TestConfiguration"], new_test_entry["TestConfiguration"])) {
              match_test_index = static_cast<int>(entry_index);
            }
          }
          if (match_test_index == -1) {
            database[hostname][machine_index]["Tests"][new_test_entry_name].push_back(new_test_entry);
            return_value = PerfTestNewTestConfiguration;
          }
          else {
            bool deviation = false;
            YAML::Node old_test_entry = old_test_array[match_test_index];
            YAML::Node old_results = old_test_entry["TestResults"];
            YAML::Node new_results = new_test_entry["TestResults"];
            // Compare all entries
            for (YAML::const_iterator old_r = old_results.begin(); old_r != old_results.end(); ++old_r) {
              YAML::Node result_entry = old_r->second;
              // Finding entry with same name
              std::string result_name = old_r->first.Scalar();
              bool exists = new_results[result_name];
              if (exists) {
                std::string oldv_str = old_r->second.Scalar();
                std::string old_test_name = new_test_entry_name;
                std::ostringstream new_result_entry_name_stream;
                new_result_entry_name_stream << new_results[result_name];
                std::string new_result_data = new_result_entry_name_stream.str();

                // If it is a time or result compare numeric values with tolerance
                if((result_name.find("Time")==0) || (result_name.find("Result")==0)) {
                  ValueTolerance old_valtol(oldv_str);
                  ValueTolerance new_valtol(new_results[result_name].Scalar());
                  if(old_valtol.use_tolerance) {
                    double diff = old_valtol.value - new_valtol.value;
                    diff*=diff;

                    double normalization = old_valtol.value;
                    normalization*=normalization;
                    if(normalization==0?diff>0:diff/normalization>old_valtol.tolerance*old_valtol.tolerance) {
                      deviation = true;
                      std::cout << std::endl
                          << "DeviationA in Test: \"" << old_test_name
                          << "\" for entry \"" <<  result_name << "\"" << std::endl;
                      std::cout << "  Existing Value: \"" << oldv_str << "\"" << std::endl;
                      std::cout << "  New Value:      \"" << new_result_data << "\"" << std::endl << std::endl;
                    }
                  }
                  else {
                    if( (old_valtol.lower>new_valtol.value) || (old_valtol.upper<new_valtol.value)) {
                      deviation = true;
                      std::cout << std::endl
                          << "DeviationB in Test: \"" << old_test_name
                          << "\" for entry \"" <<  result_name << "\"" << std::endl;
                      std::cout << "  Existing Value: \"" << oldv_str << "\"" << std::endl;
                      std::cout << "  New Value:      \"" << new_result_data << "\"" << std::endl << std::endl;
                    }
                  }
                }
                else {
                  // Compare exact match for every other type of entry
                  if(oldv_str.compare(new_result_data)!=0) {
                    deviation = true;
                    std::cout << std::endl
                        << "DeviationC in Test: \"" << old_test_name
                        << "\" for entry \"" <<  result_name << "\"" << std::endl;
                    std::cout << "  Existing Value: \"" << oldv_str << "\"" << std::endl;
                    std::cout << "  New Value:      \"" << new_result_data << "\"" << std::endl << std::endl;
                  }
                }
              }
              // An old value was not given in the new test: this is an error;
              if(!exists) {
                std::cout << "Error New test has same name as an existing one, but one of the old entries is missing." << std::endl;
                deviation = true;
              }
            }
            if(deviation) {
              return_value = PerfTestFailed;
            }
            else {
              // Did someone add new values to the test?
              if(new_results.size()!=old_results.size()) {
                for (YAML::const_iterator new_r = new_results.begin(); new_r != new_results.end(); ++new_r) {
                  if(!old_results[new_r->first.Scalar()]) {
                    old_results[new_r->first.Scalar()] = (new_r->second);
                  }
                }
                return_value = PerfTestUpdatedTest;
              }
            }
          }
        }
        else { // End Test Exists
          // Add new test if no match was found
          database[hostname][machine_index]["Tests"][new_test_entry_name].push_back(new_test_entry);
          return_value = PerfTestNewTest;
        }
      } // End MachineConfiguration Exists
    } // End loop over MachineConfigurations

    // Did not find matching MachineConfiguration
    if(is_new_config) {
      YAML::Node machine_entry;
      machine_entry["MachineConfiguration"] = machine_config;
      machine_entry["Tests"][new_test_entry_name].push_back(new_test_entry);
      database[hostname].push_back(machine_entry);
      return_value = PerfTestNewConfiguration;
    }
  }
  else { // Machine Entry does not exist
    YAML::Node machine_entry;
    machine_entry["MachineConfiguration"] = machine_config;
    machine_entry["Tests"][new_test_entry_name].push_back(new_test_entry);
    database[hostname].push_back(machine_entry);
    return_value = PerfTestNewMachine;
  }

  if(return_value>PerfTestPassed) {
    std::ofstream fout(filename.c_str());
    fout << database << std::endl;
  }
  return return_value;
}

} // namespace Kokkos
