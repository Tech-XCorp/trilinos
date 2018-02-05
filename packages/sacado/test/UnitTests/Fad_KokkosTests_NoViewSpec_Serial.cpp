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
#include <ostream>

#include "Kokkos_Parallel.hpp"
#include "Kokkos_Serial.hpp"
#include "Kokkos_View.hpp"
#include "Sacado_Fad_Ops.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_Ptr.hpp"
#include "Teuchos_UnitTestRepository.hpp"
#include "Teuchos_toString.hpp"

// Disable view specializations
#define SACADO_DISABLE_FAD_VIEW_SPEC

#include "Fad_KokkosTests.hpp"

namespace Kokkos {
struct LayoutLeft;
struct LayoutRight;
}  // namespace Kokkos

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
