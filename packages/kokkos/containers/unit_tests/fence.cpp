#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    typedef Kokkos::DefaultExecutionSpace exec_space;
    typedef Kokkos::View<int*> uvm_t;
    typedef Kokkos::View<int*, Kokkos::HostSpace> hst_t;
    const int N = 10;

    Kokkos::DualView<int*> dual("dual", N);
    dual.modify_device(); // so subsequent sync_host will fence
    auto dual_h = dual.view_host();
    dual.sync_host(); // if this was moved after the deep_copy we'd be ok

    uvm_t a("uvm a", N);
    uvm_t b(hst_t("hst", N).data(), N);

    Kokkos::fence(); // just to emphasize the deep_copy below is the issue
    Kokkos::deep_copy(exec_space(), a, b);

   // dual.sync_host(); // this will work on Kepler ONLY if sync_host NOT called earlier

   printf("Read dual on host: %d\n", dual_h(0));        
  }
  Kokkos::finalize();
  return 0;
}

/*
  const int N = 1000;
  using exec_space = Kokkos::DefaultExecutionSpace;
  typedef Kokkos::View<int*> uvm_t;
  typedef Kokkos::View<int*, Kokkos::HostSpace> hst_t;
  uvm_t uvm1("a", N);
  hst_t hst("b", N);
  uvm_t uvmA("a", N);
  uvm_t uvm2(hst.data(), hst.size());
  Kokkos::fence();
  Kokkos::deep_copy(exec_space(), uvm1, uvm2);
  uvmA(1) = 1;  // only gives a bus error with this line included


}
*/

