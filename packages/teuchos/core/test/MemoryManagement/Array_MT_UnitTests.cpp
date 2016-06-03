/*
 * Array_MT_UnitTests.cpp
 *
 *  Created on: May 6, 2016
 *      Author: micheldemessieres
 */

#include "TeuchosCore_ConfigDefs.hpp"

#ifdef HAVE_TEUCHOSCORE_CXX11

//#define REMOVE_MUTEX_LOCK_FOR_ARRAY // adding this line will remove the mutex lock in Array and cause the mtArrayMultipleReads unit test to fail

#include "General_MT_UnitTests.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include <vector>
#include <thread>

/*
 
 Array Notes
 
 The member variables are the first target for thread safe investigation
 
 #ifdef HAVE_TEUCHOS_ARRAY_BOUNDSCHECK
   RCP<std::vector<T> > vec_;
   mutable ArrayRCP<T> extern_arcp_;
   mutable ArrayRCP<const T> extern_carcp_;
 #else
   std::vector<T> vec_;
 #endif
 
 inline std::vector<T>& vec(
   bool isStructureBeingModified = false,
   bool activeIter = false
 );
 
 
 vec_ (HAVE_TEUCHOS_ARRAY_BOUNDSCHECK version):
   This is a strong RCP which is set in the constructor and never changed, though it can be accessed - it should be fine for all thread operations.
   Multiple threads should be safe to read vec_ which will generate the same behavior as reading a std::vector
   We cannot have simultaneous writing but that is the restriction of our data acess
 
 vec_ (release version):
   This seems to be ok - simultaneous manipulations of array memory will be undefined behavior and should be avoided by design
   vec_ is set in the constructors so should be fine
 
 extern_arcp_ and extern_carcp_ (HAVE_TEUCHOS_ARRAY_BOUNDSCHECK only):
   This is defined in begin() and therefore needs protection - resolved with a mutex lock during the check and allocation
 
 inline std::vector<T>& vec() (HAVE_TEUCHOS_ARRAY_BOUNDSCHECK version)
   In
 
 inline std::vector<T>& vec() (release version)
   In this case, vec() is simply an inline return on vec_ so it is ok - it is equivalent to vec_
 
 */
namespace {
  
  using Teuchos::Array;
  using Teuchos::RCP;
  
  static void share_array_to_threads(RCP<Array<int>> shared_array) {
    while (!ThreadTestManager::s_bAllowThreadsToRun) {}
    for (Array<int>::iterator iter = shared_array->begin(); iter < shared_array->end(); ++iter) {
      // do nothing here - the point is to call the begin()
    }
  }
  
  TEUCHOS_UNIT_TEST( Array, mtArrayMultipleReads )
  {
    // the point of this test was to validate that multiple threads can safely read an Array
    // we expected it to fail originally because the begin() call in Debug will set extern_arcp_
    // so the first strategy was to make a race condition on that allocation to demonstrate we could see this problem
    // note that begin() is a const but the internal extern_arcp_ object is mutable - that is our target here
    
    const int numThreads = 4;
    const int numTests = 100;
    
    for (int testCycle = 0; testCycle < numTests; ++testCycle) {
      try {
        std::vector<std::thread> threads;
        ThreadTestManager::s_bAllowThreadsToRun = false;
        
        RCP<Array<int>> array_rcp = rcp(new Array<int>( 10, 3 )); // makes an array of length 1000 with each element set to 3
        
        for (int i = 0; i < numThreads; ++i) {
          threads.push_back( std::thread(share_array_to_threads, array_rcp) );
        }
        
        ThreadTestManager::s_bAllowThreadsToRun = true;     // let the threads run
        for (int i = 0; i < threads.size(); ++i) {
          threads[i].join();
        }
      }
      TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);
    }

    TEST_EQUALITY_CONST(0, 0);
  }
  
} // end namespace

#endif // HAVE_TEUCHOSCORE_CXX11



