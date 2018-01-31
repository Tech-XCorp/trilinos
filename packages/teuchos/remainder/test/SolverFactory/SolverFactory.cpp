
#include <ostream>

#include "Teuchos_FancyOStream.hpp"
#include "Teuchos_UnitTestHelpers.hpp"

namespace { // (anonymous)

  // For now, stub unit test.
  // This will change once we add SolverFactory.
  TEUCHOS_UNIT_TEST( SolverFactory, Test0 )
  {
    using std::endl;

    out << "SolverFactory Test0" << endl;
    success = true;
  }
} // (anonymous)

