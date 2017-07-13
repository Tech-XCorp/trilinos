// @HEADER
// ***********************************************************************
//
//                    Kokkos: Common Tools Package
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

#ifndef KOKKOS_YAMLPERFTESTARCHIVE_HPP
#define KOKKOS_YAMLPERFTESTARCHIVE_HPP

#include "yaml-cpp/yaml.h"
#include <sstream>

#if defined __clang__
  #define KOKKOS_COMPILER_NAME "Clang"
  #define KOKKOS_COMPILER_VERSION __clang_major__*100+__clang_minor__*10+__clang_patchlevel__
  #define KOKKOS_COMPILER_CLANG 1
#endif

namespace Kokkos {
  /**
   * \brief ValueTolerance is a struct to keep a tuple of value and a tolerance.
   * The tolerance can be either expressed as a relative or through an upper and
   * lower bound.
   */
struct ValueTolerance {
  double value;
  double lower;
  double upper;
  double tolerance;
  bool use_tolerance;

  ValueTolerance();
  ValueTolerance(double val, double tol);
  ValueTolerance(double val, double low, double up);

  ValueTolerance(std::string str);

  bool operator ==(ValueTolerance& rhs);

  std::string as_string();
  void from_string(const std::string& valtol_str);
};

/**
 * \class YAMLTestNode
 * \brief 
 *
 * This subclass of YAMLObject generates an YAML list in a style more
 * suitable for a performance test archive. It also provides a number
 * of convenience functions helpful for working with a test archive.
 */

class YAMLTestNode {
  public:
    YAMLTestNode() {
    }

    YAMLTestNode(const std::string &tag) {
    }
    
    void addDouble (const std::string& name, double val);
    void addInt (const std::string& name, int val);
    void addBool (const std::string& name, bool val);
    void addValueTolerance(const std::string& name, ValueTolerance val);
    void addString (const std::string& name, std::string val);

    template<class T>
    void addAttribute (const std::string& name, T val) {
      for (size_t i = 0; i < name.length (); i++) {
        if (name[i] == ' ') {
          return;
        }
      }
      std::ostringstream strs;
      strs << val;
      YAMLTestNode entry (name);
      entry.addContent (strs.str ());
      YAMLTestNode::addChild (entry);
    }
    
    bool isEmpty() const {
      return true;
    }
    void addContent(const std::string& contentLine) {
      throw std::logic_error( "Not implemented" );
    }   
    void addChild(const YAMLTestNode& child) {
      throw std::logic_error( "Not implemented" );
    }

    YAMLTestNode getChild(int i) const;

    YAMLTestNode getChild(const std::string &name) const;

    const std::string& getTag() const
    {
      throw std::logic_error( "Not implemented" );
    }
    int numContentLines() const
    {
      throw std::logic_error( "Not implemented" );
    }
    const std::string& getContentLine(int i) const
    {
      throw std::logic_error( "Not implemented" );
    }
    int numChildren() const {
      throw std::logic_error( "Not implemented" );    
    }

    bool hasChild(const std::string &name) const;

    void appendContentLine(const size_t& i, const std::string &str);

    bool hasSameElements(YAMLTestNode const & lhs) const;
    
    void loadYAMLFile(const std::string &fileName) {
      nodes = YAML::LoadAllFromFile(fileName); // get all the nodes
    }

  private:
    std::vector<YAML::Node> nodes;
};

/**
 * \brief PerfTest_MachineConfig generates a basic machine configuration XMLTestNode.
 *
 * \details The function provides a starting point for a machine configuration. Users
 * should add new entries to the returned XMLTestNode to provide test relevant machine
 * configuration entries. For example Kokkos users might want to provide the name of the
 * user Kokkos NodeType or Kokkos DeviceType. The returned config contains information
 * mostly extracted from /proc/cpuinfo if possible. On non unix systems most values
 * will be unknown. Entries are:
 * - Compiler: The compiler name.
 * - Compiler_Version: A compiler version number.
 * - CPU_Name: The CPUs model name.
 * - CPU_Sockets: Number of CPU sockets in the system.
 * - CPU_Cores_Per_Socket: Number of CPU cores per socket.
 * - CPU_Total_HyperThreads: Total number of threads in a node.
 */
YAMLTestNode PerfTest_MachineConfig();

/**
 * \brief ReturnValues for PerfTest_CheckOrAdd_Test
 */
enum PerfTestResult {PerfTestFailed, PerfTestPassed,
                     PerfTestNewMachine, PerfTestNewConfiguration,
                     PerfTestNewTest, PerfTestNewTestConfiguration,
                     PerfTestUpdatedTest};

/**
 * \brief Check whether a test is present and match an existing test
 *   in an archive.
 *
 * This function consumes a machine configuration XMLTestNode and a
 * test entry XMLTestNode.  It will attempt to read from an existing
 * file containing a test archive, or generate a new one.  Optionally
 * a hostname override can be provided, which is for example useful
 * when running on clusters, where the cluster name should be used for
 * the test entries instead of the compute node name.
 * PerfTest_CheckOrAdd_Test will go through the test archive and
 * search for a matching machine name with matching machine
 * configuration and matching test configuration. If one is found the
 * result values will be compared, if not a new test entry is
 * generated and the result written back to the file.
 *
 * \param machine_config [in] An XMLTestNode describing the machine
 *   configuration.
 * \param new_test [in] An XMLTestNode describing the test.
 * \param filename [in] The name of a file containing a performance
 *   test archive.
 * \param ext_hostname [in] An optional hostname to be used instead of
 *   the one provided by the OS.
 *
 * \return Whether a matching test is found, or if it was added to an
 *   archive.
 *
 * Here is the list of valid return values:
 *
 * - PerfTestFailed: Matching configuration found, but results are
 *   deviating more than the allowed tolerance.
 * - PerfTestPassed: Matching configuration found, and results are
 *   within tolerances.
 * - PerfTestNewMachine: The test archive didn't contain an entry with
 *   the same machine name. A new entry was generated.
 * - PerfTestNewConfiguration: No matching machine configuration was
 *   found. A new entry was generated.
 * - PerfTestNewTest: No matching testname was found. A new entry was
 *   generated.
 * - PerfTestNewTestConfiguration: A matching testname was found, but
 *   different parameters were used. A new entry was generated.
 * - PerfTestUpdatedTest: A matching test was found but more result
 *   values were given then previously found. The entry is updated.
 *   This will only happen if all the old result values are present in
 *   the new ones, and are within their respective tolerances.
 */
PerfTestResult
PerfTest_CheckOrAdd_Test (YAMLTestNode machine_config,
                          YAMLTestNode new_test,
                          const std::string filename,
                          const std::string ext_hostname = std::string ());

} // namespace Kokkos

#endif
