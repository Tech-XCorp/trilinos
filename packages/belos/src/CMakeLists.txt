

######################################################################
## Package-specific configuration options
######################################################################

TRIBITS_CONFIGURE_FILE(${PACKAGE_NAME}_config.h)

######################################################################
## Define the header and source files (and directories)
######################################################################

SET(HEADERS "")
SET(SOURCES "")

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

SET(HEADERS ${HEADERS}
  ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.h
  )

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

# CORE
APPEND_SET(HEADERS
  Belos_Details_EBelosSolverType.hpp
  BelosBiCGStabIter.hpp
  BelosBiCGStabSolMgr.hpp
  BelosBlockCGIter.hpp
  BelosBlockCGSolMgr.hpp
  BelosBlockFGmresIter.hpp
  BelosBlockGmresIter.hpp
  BelosBlockGmresSolMgr.hpp
  BelosCGIteration.hpp
  BelosCGIter.hpp
  BelosCGSingleRedIter.hpp
  BelosConfigDefs.hpp
  BelosDGKSOrthoManager.hpp
  BelosFixedPointIter.hpp
  BelosFixedPointIteration.hpp
  BelosFixedPointSolMgr.hpp
  BelosGCRODRIter.hpp
  BelosGCRODRSolMgr.hpp
  BelosGmresIteration.hpp
  BelosGmresPolyOp.hpp
  BelosGmresPolySolMgr.hpp
  BelosICGSOrthoManager.hpp
  BelosIMGSOrthoManager.hpp
  BelosIteration.hpp
  BelosLinearProblem.hpp
  BelosLSQRIter.hpp
  BelosLSQRIteration.hpp
  BelosLSQRSolMgr.hpp
  BelosLSQRStatusTest.hpp
  BelosMatOrthoManager.hpp
  BelosMinresIteration.hpp
  BelosMinresIter.hpp
  BelosMinresSolMgr.hpp
  BelosMultiVec.hpp
  BelosMultiVecTraits.hpp
  BelosMVOPTester.hpp
  BelosOperator.hpp
  BelosOperatorTraits.hpp
  BelosOrthoManagerFactory.hpp
  BelosOrthoManager.hpp
  BelosOrthoManagerTest.hpp
  BelosOutputManager.hpp
  BelosPCPGIter.hpp
  BelosPCPGSolMgr.hpp
  BelosPseudoBlockCGIter.hpp
  BelosPseudoBlockCGSolMgr.hpp
  BelosPseudoBlockGmresIter.hpp
  BelosPseudoBlockGmresSolMgr.hpp
  BelosPseudoBlockStochasticCGIter.hpp
  BelosPseudoBlockStochasticCGSolMgr.hpp
  BelosRCGIter.hpp
  BelosRCGSolMgr.hpp
  BelosSimpleOrthoManager.hpp
  BelosSolverManager.hpp
  BelosSolverFactory_decl.hpp
  BelosSolverFactory_impl.hpp
  BelosCustomSolverFactory.hpp
  BelosStatusTestCombo.hpp
  BelosStatusTestFactory.hpp
  BelosStatusTestGeneralOutput.hpp
  BelosStatusTestGenResNorm.hpp
  BelosStatusTestGenResSubNorm.hpp
  BelosStatusTest.hpp
  BelosStatusTestImpResNorm.hpp
  BelosStatusTestLogResNorm.hpp
  BelosStatusTestMaxIters.hpp
  BelosStatusTestOutputFactory.hpp
  BelosStatusTestOutput.hpp
  BelosStatusTestResNorm.hpp
  BelosStatusTestResNormOutput.hpp
  BelosStatusTestUserOutput.hpp
  BelosStochasticCGIteration.hpp
  BelosStubTsqrAdapter.hpp
  BelosTFQMRIter.hpp
  BelosTFQMRSolMgr.hpp
  BelosPseudoBlockTFQMRIter.hpp
  BelosPseudoBlockTFQMRSolMgr.hpp
  BelosTypes.hpp
  Belos_Details_LinearSolver.hpp
  Belos_Details_LinearSolverFactory.hpp
  Belos_Details_registerLinearSolverFactory.hpp
  BelosUtils.hpp
  )

# UTIL
APPEND_SET(SOURCES
  Belos_Details_EBelosSolverType.cpp
  BelosTypes.cpp
  BelosVersion.cpp
  Belos_Details_registerLinearSolverFactory.cpp
  BelosSolverFactory.cpp
  )

# COMP_H
APPEND_SET(HEADERS
  BelosBlockCGSolMgr.hpp
  BelosMinresSolMgr.hpp
  BelosBlockGmresSolMgr.hpp
  BelosPseudoBlockCGSolMgr.hpp
  BelosPseudoBlockGmresSolMgr.hpp
  BelosPseudoBlockStochasticCGSolMgr.hpp
  BelosTFQMRSolMgr.hpp
  BelosPseudoBlockTFQMRSolMgr.hpp
  BelosCGIter.hpp
  BelosCGSingleRedIter.hpp
  BelosBlockCGIter.hpp
  BelosMinresIter.hpp
  BelosBlockGmresIter.hpp
  BelosPseudoBlockCGIter.hpp
  BelosPseudoBlockGmresIter.hpp
  BelosPseudoBlockStochasticCGIter.hpp
  BelosTFQMRIter.hpp
  BelosPseudoBlockTFQMRIter.hpp
  )

######################################################################
## Add files to HEADERS and /or SOURCES, depending on the values of
## certain configure-time options.
######################################################################

ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_TSQR)
IF(${PACKAGE_NAME}_ENABLE_TSQR)
  #
  # Build the TSQR-based OrthoManager and MatOrthoManager classes.
  #
  APPEND_SET(HEADERS BelosTsqrOrthoManagerImpl.hpp BelosTsqrOrthoManager.hpp)
ENDIF() # ${PACKAGE_NAME}_ENABLE_TSQR

#
# See the comments on "experimental" code in belos/CMakeLists.txt.  In
# particular, experimental headers and source files are not included
# in the build or install unless the Belos_ENABLE_Experimental CMake
# option was set at configure time.
#
APPEND_SET(${PACKAGE_NAME}_EXPERIMENTAL_HEADERS
  BelosBlockGCRODRIter.hpp
  BelosBlockGCRODRSolMgr.hpp
  BelosInnerSolveResult.hpp
  BelosInnerSolver.hpp
  BelosProjectedLeastSquaresSolver.hpp
)

APPEND_SET(${PACKAGE_NAME}_EXPERIMENTAL_SOURCES
  BelosInnerSolveResult.cpp
)

ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_Experimental)
IF(${PACKAGE_NAME}_ENABLE_Experimental)
  #
  # Add experimental files to HEADERS and SOURCES.
  #
  APPEND_SET(HEADERS ${${PACKAGE_NAME}_EXPERIMENTAL_HEADERS})
  APPEND_SET(SOURCES ${${PACKAGE_NAME}_EXPERIMENTAL_SOURCES})
ENDIF() # ${PACKAGE_NAME}_ENABLE_Experimental

######################################################################
## Create headers for ETI and instantiate
######################################################################
# glob the binary dir last to get all of the
# auto-generated ETI headers
INCLUDE(TribitsLibraryMacros)
INCLUDE(TrilinosCreateClientTemplateHeaders)
SET_AND_INC_DIRS(DIR ${CMAKE_CURRENT_SOURCE_DIR})
TRILINOS_CREATE_CLIENT_TEMPLATE_HEADERS(${DIR})
SET_AND_INC_DIRS(DIR ${CMAKE_CURRENT_BINARY_DIR})
APPEND_GLOB(HEADERS ${DIR}/*.hpp)

# TODO Add Belos_ENABLE_EXPLICIT_INSTANTIATION and switch this
# IF(${PACKAGE_NAME}_ENABLE_EXPLICIT_INSTANTIATION)
IF(${PROJECT_NAME}_ENABLE_EXPLICIT_INSTANTIATION)
  # Not sure for this yet
  # We have createValidParameterList.cpp in the doc folder
  # which needs tpetra/src/BelosTpetraAdapter.hpp
  # It might be more appropriate to just make that exe be the explicit instantiation
  # and use the decl.hpp and def.hpp includes. But that might lead to duplication eventually
  # so started by moving that to the ETI class.
  IF (${PACKAGE_NAME}_ENABLE_Tpetra)
    INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../tpetra/src )
  ENDIF()

  APPEND_SET(SOURCES Belos_ExplicitTemplateInstantiation.cpp)
  MESSAGE(STATUS "Belos ETI Enabled")
ELSE()
  MESSAGE(STATUS "Belos ETI Disabled")
ENDIF()


######################################################################
## Define the targets for the package's library/ies.
######################################################################

IF (${PACKAGE_NAME}_ENABLE_TSQR)
  # If TSQR support is enabled in Belos, we have to link Belos with
  # the appropriate Kokkos libraries, since TSQR is implemented in
  # Kokkos.
  TRIBITS_ADD_LIBRARY(
    belos
    HEADERS ${HEADERS}
    SOURCES ${SOURCES}
    )
ELSE ()
  TRIBITS_ADD_LIBRARY(
    belos
    HEADERS ${HEADERS}
    SOURCES ${SOURCES}
    )
ENDIF ()
