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
#include <cstring>
#include <cstdlib>

#include <Teuchos_XMLPerfTestArchive.hpp>

#ifndef CONVERT_YAML
#include <Teuchos_XMLObject.hpp>
#include <Teuchos_FileInputSource.hpp>
#endif


#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#include <Winsock2.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <unistd.h>
#endif

namespace Teuchos {

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

#ifndef CONVERT_YAML
  XMLTestNode::XMLTestNode():XMLObject() {}
  XMLTestNode::XMLTestNode(const std::string &tag):XMLObject(tag) {}
  XMLTestNode::XMLTestNode(XMLObjectImplem *ptr):XMLObject(ptr) {}
  XMLTestNode::XMLTestNode(XMLObject obj):XMLObject(obj) {}
  void  XMLTestNode::addDouble (const std::string &name, double val) {
    addAttribute<double>(name,val);
  }
  void  XMLTestNode::addInt (const std::string &name, int val) {
    addAttribute<int>(name,val);
  }
  void  XMLTestNode::addBool (const std::string &name, bool val) {
    addAttribute<bool>(name,val);
  }
  void XMLTestNode::addValueTolerance(const std::string &name, ValueTolerance val){
    addAttribute<std::string>(name,val.as_string());
  }
  void  XMLTestNode::addString (const std::string &name, std::string val) {
    addAttribute<std::string>(name,val);
  }
  bool XMLTestNode::hasChild(const std::string &name) const {
    bool found = false;
    for(int i = 0; i < numChildren(); i++) {
      if(name.compare(XMLObject::getChild(i).getTag()) == 0) {
        found = true;
        i = numChildren();
      }
    }
    return found;
  }
  void XMLTestNode::appendContentLine(const size_t& i, const std::string &str) {
    ptr_->appendContentLine(i,str);
  }
  XMLTestNode XMLTestNode::getChild(const std::string &name) const {
    XMLTestNode child;
    for(int i = 0; i < numChildren(); i++) {
      if(name.compare(XMLObject::getChild(i).getTag()) == 0)
        child = XMLObject::getChild(i);
    }
    return child;
  }
  XMLTestNode XMLTestNode::getChild(const int &i) const {
    return XMLObject::getChild(i);
  }
  const XMLObject* XMLTestNode::xml_object() const {
    return (XMLObject*) this;
  }
  bool XMLTestNode::hasSameElements(XMLTestNode const & lhs) const {
    if((numChildren()!=lhs.numChildren()) ||
       (numContentLines()!= lhs.numContentLines()) ||
       (getTag().compare(lhs.getTag())!=0)) return false;

    for(int i = 0; i<numChildren(); i++) {
      const XMLTestNode child = XMLObject::getChild(i);
      if( (!lhs.hasChild(child.getTag())) ||
          (!child.hasSameElements(lhs.getChild(child.getTag()))) ) return false;
    }

    for(int i = 0; i<numContentLines(); i++)
      if(getContentLine(i).compare(lhs.getContentLine(i))!=0) return false;

    return true;
  }
#endif

#ifdef CONVERT_YAML
  YAML::Node PerfTest_MachineConfig() {
#else
  XMLTestNode PerfTest_MachineConfig() {
#endif

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

#ifdef CONVERT_YAML
  YAML::Node machine_config;
  machine_config["Compiler"] = TEUCHOS_COMPILER_NAME;
  machine_config["Compiler_Version"] = TEUCHOS_COMPILER_VERSION;
  machine_config["CPU_Name"] = cpuname;
  machine_config["CPU_Sockets"] = highest_socketid+1;
  machine_config["CPU_Cores_Per_Socket"] = cores_per_socket;
  machine_config["CPU_Total_HyperThreads"] = threads;
#else
  XMLTestNode machine_config("MachineConfiguration");
  machine_config.addString("Compiler", TEUCHOS_COMPILER_NAME);
  machine_config.addInt("Compiler_Version",  TEUCHOS_COMPILER_VERSION);
  machine_config.addString("CPU_Name", cpuname);
  machine_config.addInt("CPU_Sockets", highest_socketid+1);
  machine_config.addInt("CPU_Cores_Per_Socket", cores_per_socket);
  machine_config.addInt("CPU_Total_HyperThreads", threads);
#endif

  return machine_config;
}

#ifdef CONVERT_YAML
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
    if(!hasSameElements(sub_a, sub_b)) {
      return false;
    }
  }
  
  return true;
}
#endif

#ifdef CONVERT_YAML
PerfTestResult
PerfTest_CheckOrAdd_Test (YAML::Node machine_config,
                          YAML::Node new_test_with_name,
                          const std::string filename,
                          const std::string ext_hostname)
#else
PerfTestResult
PerfTest_CheckOrAdd_Test (XMLTestNode machine_config,
                          XMLTestNode new_test,
                          const std::string filename,
                          const std::string ext_hostname)
#endif
{
#ifdef CONVERT_YAML
  YAML::Node database;
#else
  XMLTestNode database;
#endif
  PerfTestResult return_value = PerfTestPassed;
  bool is_new_config = true;

  // Open Database File
  //
  // FIXME (mfh 09 Apr 2014) This actually opens the file twice.
  if (std::ifstream (filename.c_str ())) {
#ifdef CONVERT_YAML
    database = YAML::LoadFile(filename);
#else
    database = FileInputSource (filename).getObject ();
#endif
  }

  // Get Current Hostname
  char hostname[256];
  memset (hostname, 0, 256);
  if (ext_hostname.empty ()) {
    gethostname (hostname, 255);
  } else {
    strncat (hostname, ext_hostname.c_str (), 255);
  }


#ifdef CONVERT_YAML
  YAML::Node new_test_entry = new_test_with_name.begin()->second["TestEntry"];
#else
  XMLTestNode new_test_entry = new_test.getChild ("TestEntry");
#endif


#ifndef CONVERT_YAML
  if (database.isEmpty()) {
    database = XMLTestNode ("PerfTests");
  }
#endif

  // Does hostname exist?
#ifdef CONVERT_YAML
  if (database["PerfTests"][hostname]) {
#else
  if (database.hasChild (hostname)) {
#endif

#ifdef CONVERT_YAML
    YAML::Node machine = database["PerfTests"][hostname];
#else
    XMLTestNode machine = database.getChild (hostname);
#endif


    // Find matching machine configuration
    
#ifdef CONVERT_YAML
    for (YAML::const_iterator i = machine.begin(); i != machine.end(); ++i) {
      YAML::Node configuration = i->second;
#else
    for (int i = 0; i < machine.numChildren (); ++i) {
      XMLTestNode configuration = machine.getChild (i);
#endif
      
#ifndef CONVERT_YAML
      TEUCHOS_TEST_FOR_EXCEPTION(
        configuration.getTag ().compare ("Configuration") != 0,
        std::runtime_error, "Unexpected Tag \"" << configuration.getTag ()
        << "\"; only children with Tag = \"Configuration\" are allowed in a "
        "MachineEntry.");

      TEUCHOS_TEST_FOR_EXCEPTION(
        ! configuration.hasChild ("MachineConfiguration") ||
        ! configuration.hasChild ("Tests"),
        std::runtime_error,
        "A Configuration needs to have a child \"MachineConfiguration\" and a "
        "child \"Tests\".");
#endif

#ifdef CONVERT_YAML
      YAML::Node machine_configuration = configuration["MachineConfiguration"];
      YAML::Node old_tests = configuration["Tests"];
      if (hasSameElements(machine_configuration, machine_config)) {
#else
      XMLTestNode machine_configuration = configuration.getChild ("MachineConfiguration");
      XMLTestNode old_tests = configuration.getChild ("Tests");
      if (machine_configuration.hasSameElements (machine_config)) {
#endif
        is_new_config = false;

        // Find existing test with same tag as the new test
#ifdef CONVERT_YAML
        std::string testName = new_test_with_name.begin()->first.Scalar();
        if(old_tests[testName]) {
          YAML::Node old_test = old_tests[testName];
          YAML::Node match_test_node;
#else
        if (old_tests.hasChild (new_test.getTag ())) {
          XMLTestNode old_test = old_tests.getChild (new_test.getTag ());
          int match_test_config = -1;
#endif

#ifdef CONVERT_YAML
          for (YAML::const_iterator k = old_test.begin(); k != old_test.end(); ++k) {
            YAML::Node old_test_entry = k->second;
#else
          for (int k = 0; k < old_test.numChildren (); ++k) {
            XMLTestNode old_test_entry = old_test.getChild (k);
#endif

#ifndef CONVERT_YAML
            TEUCHOS_TEST_FOR_EXCEPTION(
              ! old_test_entry.hasChild ("TestConfiguration") ||
              ! new_test_entry.hasChild ("TestResults"),
              std::runtime_error, "A TestEntry needs to have a child "
              "\"TestConfiguration\" and a child \"TestResults\".");
#endif

#ifdef CONVERT_YAML
            if (hasSameElements(old_test_entry["TestConfiguration"], new_test_entry["TestConfiguration"])) {
              match_test_node = k->second;
            }
#else
            if (old_test_entry.getChild ("TestConfiguration").hasSameElements (new_test_entry.getChild ("TestConfiguration"))) {
              match_test_config = k;
            }
#endif
          }

#ifdef CONVERT_YAML
          if (match_test_node.IsNull()) {
            old_test.push_back(new_test_entry);
#else
          if (match_test_config < 0) {
            old_test.addChild (new_test_entry);
#endif
            return_value = PerfTestNewTestConfiguration;
          }
          else {
            bool deviation = false;
           
#ifdef CONVERT_YAML
            YAML::Node old_test_entry = match_test_node;
            YAML::Node old_results = old_test_entry["TestResults"];
            YAML::Node new_results = new_test_entry["TestResults"];
#else
            XMLTestNode old_test_entry = old_test.getChild(match_test_config);
            XMLTestNode old_results = old_test_entry.getChild ("TestResults");
            XMLTestNode new_results = new_test_entry.getChild ("TestResults");
#endif


            
#ifdef CONVERT_YAML
            // Compare all entries
            for (YAML::const_iterator old_r = old_results.begin(); old_r != old_results.end(); ++old_r) {
              YAML::Node result_entry = old_r->second;
#else
            // Compare all entries
            for (int old_r = 0; old_r < old_results.numChildren (); ++old_r) {
              XMLTestNode result_entry = old_results.getChild (old_r);
#endif
              // Finding entry with same name
#ifdef CONVERT_YAML
              std::string result_name = old_r->first.Scalar();
              bool exists = new_results[result_name];
#else
              bool exists = new_results.hasChild (result_entry.getTag ());
#endif
              if (exists) {
              
#ifdef CONVERT_YAML
                std::string oldv_str = old_r->second.Scalar();
#else
                std::string oldv_str = result_entry.getContentLine (0);
#endif

#ifdef CONVERT_YAML
                std::string old_test_name = testName;

                std::ostringstream new_result_entry_name_stream;
                new_result_entry_name_stream << new_results[result_name];
                std::string new_result_data = new_result_entry_name_stream.str();
#else
                std::string old_test_name = old_test.getTag();
                std::string result_name = result_entry.getTag();
                std::string new_result_data = new_results.getChild(result_name).getContentLine(0);
#endif

              // If it is a time or result compare numeric values with tolerance
#ifdef CONVERT_YAML
                if((result_name.find("Time")==0) || (result_name.find("Result")==0)) {
#else
                if((result_entry.getTag().find("Time")==0) || (result_entry.getTag().find("Result")==0)) {
#endif
                  ValueTolerance old_valtol(oldv_str);
                  
#ifdef CONVERT_YAML
                  ValueTolerance new_valtol(new_results[result_name].Scalar());
#else
                  ValueTolerance new_valtol(new_results.getChild(result_entry.getTag()).getContentLine(0));            
#endif
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
#ifdef CONVERT_YAML
              if(new_results.size()!=old_results.size()) {
                for (YAML::const_iterator new_r = new_results.begin(); new_r != new_results.end(); ++new_r) {
                  if(!old_results[new_r->first.Scalar()]) {
                    old_results[new_r->first.Scalar()] = (new_r->second);
                  }
                }
                return_value = PerfTestUpdatedTest;
              }
#else
              if(new_results.numChildren()!=old_results.numChildren()) {
                for(int new_r = 0; new_r < new_results.numChildren() ; new_r++) {
                  if(!old_results.hasChild(new_results.getChild(new_r).getTag())) {
                    old_results.addChild(new_results.getChild(new_r));
                  }
                }
                return_value = PerfTestUpdatedTest;
              }
#endif
            }
          }
        }
        else { // End Test Exists
          // Add new test if no match was found
#ifdef CONVERT_YAML
          old_tests.push_back(new_test_with_name);
#else
          old_tests.addChild(new_test);
#endif
          return_value = PerfTestNewTest;
        }
      } // End MachineConfiguration Exists
    } // End loop over MachineConfigurations

    // Did not find matching MachineConfiguration
    if(is_new_config) {
#ifdef CONVERT_YAML
      YAML::Node config("Configuration");
      //config.addChild(machine_config);
      //YAMLTestNode tests("Tests");
#else
      XMLTestNode config("Configuration");
      config.addChild(machine_config);
      XMLTestNode tests("Tests");
      tests.addChild(new_test);
      config.addChild(tests);
      machine.addChild(config);
#endif
      return_value = PerfTestNewConfiguration;
    }
  }
  else { // Machine Entry does not exist
#ifdef CONVERT_YAML
    database["PerfTests"][hostname]["Configuration"]["MachineConfiguration"] = machine_config;
    database["PerfTests"][hostname]["Configuration"]["Tests"] = new_test_with_name;
#else
    XMLTestNode machine(hostname);
    XMLTestNode config("Configuration");
    config.addChild(machine_config);
    XMLTestNode tests("Tests");
    tests.addChild(new_test);
    config.addChild(tests);
    machine.addChild(config);
    database.addChild(machine);
#endif
    return_value = PerfTestNewMachine;
  }

  if(return_value>PerfTestPassed) {
    std::ofstream fout(filename.c_str());
    fout << database << std::endl;
  }
  return return_value;
}
}
