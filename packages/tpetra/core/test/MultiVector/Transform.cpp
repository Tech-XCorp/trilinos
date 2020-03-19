#include "Tpetra_Core.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_transform_MultiVector.hpp"

void loop_test() {
  const int N = 100000;
  Kokkos::View<double*> input("input", N);
  Kokkos::View<double*> output("output", N);
  Kokkos::parallel_for(Kokkos::RangePolicy<>(0,N), KOKKOS_LAMBDA(const size_t i) {
    output(i) = input(i) + 1.0;
  });
  Kokkos::fence();
  std::cout << "Check can read: " << input(0) << std::endl;
}

void test(bool bHostData, bool bHostTransform) {
  printf("\n\nRUNNING TEST with data on: %s    transform on: %s\n\n",
    bHostData ? "host" : "device", bHostTransform ? "host" : "device");

  using GST = Tpetra::global_size_t;
  using map_type = Tpetra::Map<>;
  using multivec_type = Tpetra::MultiVector<>;
  using GO = map_type::global_ordinal_type;
  using device_execution_space =
    typename multivec_type::device_type::execution_space;
  using LO = typename multivec_type::local_ordinal_type;
  using Tpetra::getDefaultComm;

  auto comm = getDefaultComm ();
  const auto INVALID = Teuchos::OrdinalTraits<GST>::invalid ();
  const size_t numLocal = 13;
  const size_t numVecs  = 3;
  const GO indexBase = 0;
  auto map = rcp (new map_type (INVALID, numLocal, indexBase, comm));

  multivec_type X (map, numVecs);
  multivec_type Y (map, numVecs);

  constexpr double flagValue = -1.0;

  X.putScalar (flagValue);
  Y.putScalar (418.0);

  if(bHostData) {
    Y.sync_host ();
  }

  printf("BEFORE TRANSFORM: data Y need_sync_host() is: %s   X need_sync_host() is: %s\n",
    Y.need_sync_host() ? "true" : "false", X.need_sync_host() ? "true" : "false" );
  printf("BEFORE TRANSFORM: data Y need_sync_device() is: %s   X need_sync_device() is: %s\n",
    Y.need_sync_device() ? "true" : "false", X.need_sync_device() ? "true" : "false" );



  if(bHostTransform) {
    printf("Calling Transform on host\n");
    Tpetra::transform ("419 -> 777", Kokkos::DefaultHostExecutionSpace(), Y, X,
      KOKKOS_LAMBDA (const double& X_ij) { return X_ij + 359.0; });
  }
  else {
    printf("Calling Transform on device\n");
    Tpetra::transform ("419 -> 777", Y, X,
      KOKKOS_LAMBDA (const double& X_ij) { return X_ij + 359.0; });
  }

  printf("AFTER TRANSFORM: data Y need_sync_host() is: %s   X need_sync_host() is: %s\n",
    Y.need_sync_host() ? "true" : "false", X.need_sync_host() ? "true" : "false" );
  printf("AFTER TRANSFORM: data X need_sync_device() is: %s   X need_sync_device() is: %s\n",
    Y.need_sync_device() ? "true" : "false", X.need_sync_device() ? "true" : "false" );

  // Kokkos::fence();

  Y.sync_host ();

  auto Y_lcl = Y.getLocalViewHost ();
  std::cout << "Check for bus error on read: " << Y_lcl(0,0) << std::endl;
}

int
main (int argc, char* argv[]) {

  Tpetra::ScopeGuard tpetraScope (&argc, &argv);

  loop_test();


//  test(true, true);   // ok
//  test(false, false); // ok
//  test(false, true);  // ok
//  test(true, false);  // bus error on last line which reads Y
  return 0;
}
