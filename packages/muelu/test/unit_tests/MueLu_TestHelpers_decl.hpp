// @HEADER
//
// ***********************************************************************
//
//        MueLu: A package for multigrid based preconditioning
//                  Copyright 2012 Sandia Corporation
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
// Questions? Contact
//                    Jonathan Hu       (jhu@sandia.gov)
//                    Andrey Prokopenko (aprokop@sandia.gov)
//                    Ray Tuminaro      (rstumin@sandia.gov)
//
// ***********************************************************************
//
// @HEADER
#ifndef MUELU_TEST_HELPERS_DECL_H
#define MUELU_TEST_HELPERS_DECL_H

#include <stdio.h> //DEBUG
#include <string>
#include <set>
#ifndef _MSC_VER
#include <dirent.h>
#endif

// Teuchos
#include "Teuchos_Comm.hpp"
#include "Teuchos_OrdinalTraits.hpp"
#include "Teuchos_ScalarTraits.hpp"

#ifdef HAVE_MUELU_EPETRA
#include "Epetra_config.h"
#endif

// Xpetra
#include "Xpetra_ConfigDefs.hpp"
#include "Xpetra_DefaultPlatform.hpp"
#include "Xpetra_Parameters.hpp"
#include "Xpetra_Map.hpp"
#include "Xpetra_MapFactory.hpp"
#include "Xpetra_CrsMatrixWrap.hpp"
#include "Xpetra_CrsGraph.hpp"

// MueLu
#include "MueLu_ConfigDefs.hpp"
#include "MueLu_Exceptions.hpp"
#include "MueLu_Hierarchy.hpp"
#include "MueLu_FactoryManagerBase.hpp"
#include "MueLu_FactoryManager.hpp"

#include "MueLu_IfpackSmoother.hpp"
#include "MueLu_Level.hpp"

// Galeri
#include "Galeri_XpetraProblemFactory.hpp"
#include "Galeri_XpetraMatrixTypes.hpp"

#include "MueLu_NoFactory.hpp"

// Conditional Tpetra stuff
#ifdef HAVE_MUELU_TPETRA
#include "TpetraCore_config.h"
#include "Xpetra_TpetraCrsGraph.hpp"
#include "Xpetra_TpetraRowMatrix.hpp"
#include "Xpetra_TpetraBlockCrsMatrix.hpp"
#include "Tpetra_CrsGraph.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_Experimental_BlockCrsMatrix.hpp"
#endif

#include <MueLu_TestHelpers_Common.hpp>

namespace MueLuTests {
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ArrayRCP;
  using Teuchos::ArrayView;
  using Teuchos::arcp;
  using Teuchos::arcpFromArrayView;
  using Teuchos::rcpFromRef;
  using Teuchos::null;
  using Teuchos::arcp_reinterpret_cast;
  using Teuchos::Array;
  using Teuchos::rcp_dynamic_cast;
  using Teuchos::rcp_implicit_cast;
  using Teuchos::rcpFromRef;

  namespace TestHelpers {

    using Xpetra::global_size_t;

    class Parameters {

    private:
      Parameters() {} // static class

    public:

      static Xpetra::Parameters xpetraParameters;

      inline static RCP<const Teuchos::Comm<int> > getDefaultComm() {
        return Xpetra::DefaultPlatform::getDefaultPlatform().getComm();
      }

      inline static Xpetra::UnderlyingLib getLib() {
        return TestHelpers::Parameters::xpetraParameters.GetLib();
      }
    };

    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    class TestFactory {
#include "MueLu_UseShortNames.hpp"

    private:
      TestFactory() {} // static class

    public:

      //
      // Method that creates a map containing a specified number of local elements per process.
      //
      static const RCP<const Map> BuildMap(LO numElementsPerProc); // BuildMap()

      // Create a matrix as specified by parameter list options
      static RCP<Matrix> BuildMatrix(Teuchos::ParameterList &matrixList, Xpetra::UnderlyingLib lib);

      // Create a tridiagonal matrix (stencil = [b,a,c]) with the specified number of rows
      // dofMap: row map of matrix
      static RCP<Matrix> BuildTridiag(RCP<const Map> dofMap, Scalar a, Scalar b, Scalar c, Xpetra::UnderlyingLib lib=Xpetra::NotSpecified);

      // Create a 1D Poisson matrix with the specified number of rows
      // nx: global number of rows
      static RCP<Matrix> Build1DPoisson(GO nx, Xpetra::UnderlyingLib lib=Xpetra::NotSpecified);

      // Create a 2D Poisson matrix with the specified number of rows
      // nx: global number of rows
      // ny: global number of rows
      static RCP<Matrix> Build2DPoisson(GO nx, GO ny=-1, Xpetra::UnderlyingLib lib=Xpetra::NotSpecified);

      // Xpetra version of CreateMap
      static RCP<Map> BuildMap(Xpetra::UnderlyingLib lib, const std::set<GlobalOrdinal>& gids, Teuchos::RCP<const Teuchos::Comm<int> > comm);

      // Xpetra version of SplitMap
      static Teuchos::RCP<Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node> > SplitMap(Xpetra::UnderlyingLib lib, const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node> & Amap, const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node> & Agiven);

      static Teuchos::RCP<Xpetra::BlockedCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > CreateBlockDiagonalExampleMatrix(Xpetra::UnderlyingLib lib, int noBlocks, Teuchos::RCP<const Teuchos::Comm<int> > comm);

      static Teuchos::RCP<Xpetra::BlockedCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > CreateBlockDiagonalExampleMatrixThyra(Xpetra::UnderlyingLib lib, int noBlocks, Teuchos::RCP<const Teuchos::Comm<int> > comm);

      static Teuchos::RCP<Xpetra::BlockedCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > CreateBlocked3x3MatrixThyra(const Teuchos::Comm<int>& comm, Xpetra::UnderlyingLib lib);

      // Needed to initialize correctly a level used for testing SingleLevel factory Build() methods.
      // This method initializes LevelID and linked list of level
      static void createSingleLevelHierarchy(Level& currentLevel);

      // Needed to initialize correctly levels used for testing TwoLevel factory Build() methods.
      // This method initializes LevelID and linked list of level
      static void createTwoLevelHierarchy(Level& fineLevel, Level& coarseLevel);

      static RCP<SmootherPrototype> createSmootherPrototype(const std::string& type="Gauss-Seidel", LO sweeps=1);

    }; // class TestFactory


    // Helper class which has some Tpetra specific code inside
    // We put this into an extra helper class as we need partial specializations and
    // do not want to introduce partial specializations for the full TestFactory class
    //
    // The BuildBlockMatrix is only available with Teptra. However, if both Epetra
    // and Tpetra are enabled it may be that Tpetra is not instantiated on either
    // GO=int/long long and/or Node=Serial/OpenMP. We need partial specializations
    // with an empty BuildBlockMatrix routine for all instantiations Teptra is not
    // enabled for, but are existing in Xpetra due to Epetra enabled.
    template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
    class TpetraTestFactory {
#include "MueLu_UseShortNames.hpp"
    public:

      // Create a matrix as specified by parameter list options
      static RCP<Matrix> BuildBlockMatrix(Teuchos::ParameterList &matrixList, Xpetra::UnderlyingLib lib);

    private:
      TpetraTestFactory() {} // static class

    }; // class TpetraTestFactory

    // TAW: 3/14/2016: If both Epetra and Tpetra are enabled we need partial specializations
    //                 on GO=int/long long as well as NO=EpetraNode to disable BuildBlockMatrix
#ifdef HAVE_MUELU_EPETRA
    // partial specializations (GO=int not enabled with Tpetra)
#if !defined(HAVE_TPETRA_INST_INT_INT)
    template <class Scalar, class LocalOrdinal, class Node>
    class TpetraTestFactory<Scalar, LocalOrdinal, int, Node> {
      typedef int GlobalOrdinal;
#include "MueLu_UseShortNames.hpp"
    public:
      static RCP<Matrix> BuildBlockMatrix(Teuchos::ParameterList &matrixList, Xpetra::UnderlyingLib lib);
    private:
      TpetraTestFactory() {} // static class
    }; // class TpetraTestFactory
#endif

    // partial specializations (GO=long long not enabled with Tpetra)
#if !defined(HAVE_TPETRA_INST_INT_LONG_LONG)
    template <class Scalar, class LocalOrdinal, class Node>
    class TpetraTestFactory<Scalar, LocalOrdinal, long long, Node> {
      typedef long long GlobalOrdinal;
#include "MueLu_UseShortNames.hpp"
    public:
      static RCP<Matrix> BuildBlockMatrix(Teuchos::ParameterList &matrixList, Xpetra::UnderlyingLib lib);
    private:
      TpetraTestFactory() {} // static class
    }; // class TpetraTestFactory
#endif

    // partial specializations (NO=EpetraNode not enabled with Tpetra)
#if ((defined(EPETRA_HAVE_OMP) && !(defined(HAVE_TPETRA_INST_OPENMP))) || \
    (!defined(EPETRA_HAVE_OMP) && !(defined(HAVE_TPETRA_INST_SERIAL))))

    template <class Scalar, class LocalOrdinal, class GlobalOrdinal>
    class TpetraTestFactory<Scalar, LocalOrdinal, GlobalOrdinal, Xpetra::EpetraNode> {
      typedef Xpetra::EpetraNode Node;
#include "MueLu_UseShortNames.hpp"
    public:
      static RCP<Matrix> BuildBlockMatrix(Teuchos::ParameterList &matrixList, Xpetra::UnderlyingLib lib);
    private:
      TpetraTestFactory() {} // static class
    }; // class TpetraTestFactory
#endif
#endif // endif HAVE_MUELU_EPETRA

    //! Return the list of files in the directory. Only files that are matching '*filter*' are returned.
    ArrayRCP<std::string> GetFileList(const std::string & dirPath, const std::string & filter);




  } // namespace TestHelpers



} // namespace MueLuTests


// Macro to skip a test when UnderlyingLib==Epetra or Tpetra
#define MUELU_TEST_ONLY_FOR(UnderlyingLib) \
  if (TestHelpers::Parameters::getLib() != UnderlyingLib) { \
    out << "Skipping test for " << ((TestHelpers::Parameters::getLib()==Xpetra::UseEpetra) ? "Epetra" : "Tpetra") << std::endl; \
    return; \
  }

// Macro to skip a test when Epetra is used with Ordinal != int
#define MUELU_TEST_EPETRA_ONLY_FOR_INT(LocalOrdinal, GlobalOrdinal) \
  if (!(TestHelpers::Parameters::getLib() == Xpetra::UseEpetra && (Teuchos::OrdinalTraits<LocalOrdinal>::name() != string("int") || Teuchos::OrdinalTraits<GlobalOrdinal>::name() != string("int"))))

// Macro to skip a test when Epetra is used with Scalar != double or Ordinal != int
#define MUELU_TEST_EPETRA_ONLY_FOR_DOUBLE_AND_INT(Scalar, LocalOrdinal, GlobalOrdinal) \
  if (!(TestHelpers::Parameters::getLib() == Xpetra::UseEpetra && Teuchos::ScalarTraits<Scalar>::name() != string("double"))) \
    MUELU_TEST_EPETRA_ONLY_FOR_INT(LocalOrdinal, GlobalOrdinal)

//

//TODO: add directly to Teuchos ?
//#include "../xpetra/test/Xpetra_UnitTestHelpers.hpp" // declaration of TEUCHOS_UNIT_TEST_TEMPLATE_5_DECL


//


//! Namespace for MueLu test classes
namespace MueLuTests {

  using namespace TestHelpers;
}

#endif // ifndef MUELU_TEST_HELPERS_DECL_H
