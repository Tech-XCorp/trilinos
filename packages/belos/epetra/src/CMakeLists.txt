

#
# A) Package-specific configuration options
#

#
# B) Define the header and source files (and directories)
#

#
# src
#

SET(HEADERS "")
SET(SOURCES "")

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

#
# Core Files
#

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

APPEND_SET(HEADERS
  BelosEpetraAdapter.hpp
  BelosEpetraOperator.h
  BelosEpetraUtils.h
  Belos_Details_Epetra_registerLinearSolverFactory.hpp
  )

APPEND_SET(SOURCES
  BelosEpetraAdapter.cpp
  BelosEpetraOperator.cpp
  BelosEpetraUtils.cpp
  Belos_Details_Epetra_registerLinearSolverFactory.cpp
  )

#
# C) Handle ETI for Belos Epetra
#
# TODO Add Belos_ENABLE_EXPLICIT_INSTANTIATION and switch this
# IF(${PACKAGE_NAME}_ENABLE_EXPLICIT_INSTANTIATION)
IF(${PROJECT_NAME}_ENABLE_EXPLICIT_INSTANTIATION)
  APPEND_SET(HEADERS Belos_ExplicitTemplateInstantiation_Epetra.hpp)
  APPEND_SET(SOURCES Belos_ExplicitTemplateInstantiation_Epetra.cpp)
  MESSAGE(STATUS "Belos ETI Enabled for Epetra")
ELSE()
  MESSAGE(STATUS "Belos ETI Disabled for Epetra")
ENDIF()

#
# D) Define the targets for package's library(s)
#
TRIBITS_ADD_LIBRARY(
  belosepetra
  HEADERS ${HEADERS}
  SOURCES ${SOURCES}
  DEPLIBS belos
  )
