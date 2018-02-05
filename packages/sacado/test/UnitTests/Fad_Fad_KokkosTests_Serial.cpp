// @HEADER
// ***********************************************************************
//
//                           Sacado Package
//                 Copyright (2006) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301
// USA
// Questions? Contact David M. Gay (dmgay@sandia.gov) or Eric T. Phipps
// (etphipp@sandia.gov).
//
// ***********************************************************************
// @HEADER
#include <iosfwd>

#include <sstream>
//#include <ostream>
//#include <fstream>
#include "Kokkos_Core.hpp" // TODO: Strip to essential includes

#include "Fad_Fad_KokkosTests.hpp"
#include "KokkosExp_View_Fad.hpp"
#include "Kokkos_DynRankView.hpp"
#include "Kokkos_Layout.hpp"
#include "Kokkos_Serial.hpp"
#include "Kokkos_View.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_Ptr.hpp"
#include "Teuchos_UnitTestRepository.hpp"
#include "Teuchos_toString.hpp"

// Instantiate tests for Serial device
using Kokkos::Serial;
VIEW_FAD_TESTS_D( Serial )

int main( int argc, char* argv[] ) {
  Teuchos::GlobalMPISession mpiSession(&argc, &argv);

  // Initialize serial
  Kokkos::Serial::initialize();

  int res = Teuchos::UnitTestRepository::runUnitTestsFromMain(argc, argv);

  // Finalize serial
  Kokkos::Serial::finalize();

  return res;
}
