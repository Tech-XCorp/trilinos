/*
// @HEADER
// ***********************************************************************
//
//                    Teuchos: Common Tools Package
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
*/

#include "TeuchosCore_ConfigDefs.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_getConst.hpp"
#include "Teuchos_getBaseObjVoidPtr.hpp"
#ifdef HAVE_TEUCHOSCORE_CXX11
#  include "Teuchos_RCPStdSharedPtrConversions.hpp"
#else
#  include <mutex>
#endif

#include <vector>
#include <thread>
#include <stack>

#include "TestClasses.hpp"
#include "Teuchos_UnitTestHarness.hpp"


namespace {


using Teuchos::as;
using Teuchos::null;
using Teuchos::Ptr;
using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::rcpFromRef;
using Teuchos::rcpFromUndefRef;
using Teuchos::outArg;
using Teuchos::rcpWithEmbeddedObj;


//
// Test multi-threaded reference counting
//

static void make_large_number_of_copies(RCP<int> ptr) {
  std::vector<RCP<int> > ptrs(10000, ptr);
}

TEUCHOS_UNIT_TEST( RCP, mtRefCount )
{
  RCP<int> ptr(new int);
  TEST_ASSERT(ptr.total_count() == 1);
  std::vector<std::thread> threads;
  for (int i = 0; i < 4; ++i) {
    threads.push_back(std::thread(make_large_number_of_copies, ptr));
  }
  for (int i = 0; i < 4; ++i) {
    threads[i].join();
  }
  TEST_EQUALITY_CONST(ptr.total_count(), 1);
}

//
// Test Debug node tracing
// This test would fail with various memory problems due to the RCP node tracing not being thread safe
// This test would originally only fail with multiple threads and debug mode
// First fix is to use a single mutex to lock both RCPNodeTracer::removeRCPNode() and RCPNodeTracer::addNewRCPNode()
// Second issue can be harder to reproduce - originally we had node_->delete_obj(), followed by RCPNodeTracer::removeRCPNode()
// However if another thread calls RCPNodeTracer::addNewRCPNode() after the delete_obj() call and before the removeRCPNode() call it can claim the address space

static void create_independent_rcp_objects() {
  for(int n = 0; n < 10000; ++n ) { // note that getting the second issue to reproduce (the race condition between delete_obj() and removeRCPNode()) may not always hit in this configuration.
    RCP<int> ptr( new int ); // this allocates a new rcp ptr independent of all other rcp ptrs, and then dumps it, over and over
  }
}

TEUCHOS_UNIT_TEST( RCP, mtCreatedIndependentRCP )
{
  // first let's establish if we are in Debug mode - this particular test relies on it
  std::cout << std::endl;
  #ifdef TEUCHOS_DEBUG
	std::cout << "Running in Debug Mode. Multi-Threading originally caused the node tracing to fail in Debug." << std::endl;
  #else
	std::cout << "Running in Release Mode. Multi-Threading did not cause any problems in Release." << std::endl;
  #endif

  std::vector<std::thread> threads;
  int numThreads = 4;
  for (int i = 0; i < numThreads; ++i) {
    threads.push_back(std::thread(create_independent_rcp_objects));
  }
  for (int i = 0; i < threads.size(); ++i) {
    threads[i].join();
  }
}

// this mirrors the 'real' RCPPtrFactory case below but used std::shared_ptr instead
// The idea is to validate the logic - it follows the RCP case as closely as possible - but not that RCP stores weak and strong internally while std has weak_ptr and shared_ptr as separate entities
// Since we may not want to keep this shared_ptr case around I did not try to template this code and kept it separate
class SharedPtrFactory {
public:
	std::shared_ptr<int> getSharedPtr()
	{
		mutexForPtrAccess.lock();										// don't allow another thread to simultaneously access
		++countTotalAccessCalls;										// this is going to be the number of threads times the number of thread loops - I count it so we can verify we really did both activities (share data or make new data)
		std::shared_ptr<int> strongPtrToReturn(commonWeakPtr.lock());	// once we make the copy we secure the strong count, if it exists - but this is exactly where trouble will happen I expect - if a thread removes the last strong count right now what happens

		if(!strongPtrToReturn) { 										// if we don't have a strong count we will allocate - the data has been deleted
			strongPtrToReturn = std::shared_ptr<int>(new int); 			// actual allocation of new data
			++countNewAllocations;										// track each instance of new - let's us determine if the test really went back and forth and did proper stress testing
		}
		commonWeakPtr = strongPtrToReturn;								// reassign ourselves as a weak ptr so that we can deallocate if all threads give up strong access
		mutexForPtrAccess.unlock();										// now we can unlock
		return strongPtrToReturn; 										// return the ptr to the thread
	}

	// members are public to make the test flow convenient and keep the unit test simple - but the threads should not use the data directly - they are only allowed to have copies of the shared RCP through getRCPPtr
	std::weak_ptr<int> commonWeakPtr;									// public so we can error check at the end straight on the ptr from main - we don't want to complicate things by copying the ptr out and then accounting for that ptr's new counters
	static SharedPtrFactory sharedPtrFactory;							// the singleton factory class
	std::mutex mutexForPtrAccess;										// used to protect getRCPPtr()
	int countNewAllocations = 0;										// this counts each time we call new to make a new RCP ptr data allocation
	int countTotalAccessCalls = 0;										// this tracks all calls to getRCPPtr()
};
SharedPtrFactory SharedPtrFactory::sharedPtrFactory;

// the thread function constantly requests a std::shared_ptr<int> from the singleton class and then deletes it.
static void get_shared_ptr_from_singleton_many_times(int totalAllocations) {
	for(int n = 0; n < totalAllocations; ++n ) {
		std::shared_ptr<int> sharedPtr = SharedPtrFactory::sharedPtrFactory.getSharedPtr(); // this will constantly allocate and potentially release - move the declaration of rcpPtr out of the loop and it's a much different story.
	}
}

// In this test we have a singleton factory which shares an RCP pointer to may threads
// The key is that the singleton will dump memory when all access goes away
// So the counter will constantly go to 0 (dump memory) or from 0 to 1 (allocate with a fresh new)
// The initial failures identified by this test were intermittent so required a fairly high repeat count
TEUCHOS_UNIT_TEST( RCP, mtSharedPtrFromSingleton )
{
	std::cout << std::endl; // cosmetic

	const int sharedPtrAllocationsPerThread = 100000;
	// run multiple threads, each of which will interact with the singleton and make many copies of the RCP, which then get removed
	std::vector<std::thread> threads;
	for (int i = 0; i < 4; ++i) {
	  threads.push_back(std::thread(get_shared_ptr_from_singleton_many_times, sharedPtrAllocationsPerThread));
	}

	// join and complete all threads
	for (int i = 0; i < threads.size(); ++i) {
	  threads[i].join();
	}

	std::cout << "Total Access Calls: " << SharedPtrFactory::sharedPtrFactory.countTotalAccessCalls
			<< "	Total New Allocations: " << SharedPtrFactory::sharedPtrFactory.countNewAllocations
			<< "	Total Times We Got Shared Data: " << SharedPtrFactory::sharedPtrFactory.countTotalAccessCalls - SharedPtrFactory::sharedPtrFactory.countNewAllocations << std::endl;

	// this is a sanity check - if these numbers don't match up we are not doing what we thought
	int expectedTotalCalls = sharedPtrAllocationsPerThread * threads.size();
	TEST_EQUALITY_CONST(SharedPtrFactory::sharedPtrFactory.countTotalAccessCalls, expectedTotalCalls); // this should be 0 showing all data was deallocated

	int strongCount = SharedPtrFactory::sharedPtrFactory.commonWeakPtr.use_count();
	TEST_EQUALITY_CONST(strongCount, 0); // this should be 0 showing all data was deallocated

	const int minimumDeviationFromTrivialResult = 20; // for Multi-threading we want to establish the test is doing some minimum amount of variance but this will depend on the machine and many conditions
	if( SharedPtrFactory::sharedPtrFactory.countNewAllocations < minimumDeviationFromTrivialResult || SharedPtrFactory::sharedPtrFactory.countNewAllocations > SharedPtrFactory::sharedPtrFactory.countTotalAccessCalls - minimumDeviationFromTrivialResult ) {
		std::cout << "Multi-threading was not really testing this job: We would like to see more oscillation between sharing and new allocations and this did not hit an arbitrary threshold." << std::endl;
	}
	TEST_COMPARE( SharedPtrFactory::sharedPtrFactory.countNewAllocations, >, minimumDeviationFromTrivialResult );	// if are not actually doing new allocations it's not a good test
	TEST_COMPARE( SharedPtrFactory::sharedPtrFactory.countNewAllocations, <, SharedPtrFactory::sharedPtrFactory.countTotalAccessCalls - minimumDeviationFromTrivialResult );	// if are always doing new allocations it's also not a good test!
}

// this singleton class is used to share an RCP ptr between several threads in the unit test mtSingletonSharesRCP
class RCPFactory {
public:
	RCP<int> getRCPPtr()
	{
		mutexForPtrAccess.lock();										// don't allow another thread to simultaneously access
		++countTotalAccessCalls;										// this is going to be the number of threads times the number of thread loops - I count it so we can verify we really did both activities (share data or make new data)
		RCP<int> strongPtrToReturn = commonRCPPtr;						// once we make the copy we secure the strong count, if it exists - but this is exactly where trouble will happen I expect - if a thread removes the last strong count right now what happens
		if(strongPtrToReturn.strong_count() == 0) { 					// if we don't have a strong count we will allocate - the data has been deleted
			strongPtrToReturn = rcp(new int); 							// actual allocation of new data
			++countNewAllocations;										// track each instance of new - let's us determine if the test really went back and forth and did proper stress testing
		}
		commonRCPPtr = strongPtrToReturn.create_weak();					// reassign ourselves as a weak ptr so that we can deallocate if all threads give up strong access
		mutexForPtrAccess.unlock();										// now we can unlock
		return strongPtrToReturn; 										// return the ptr to the thread
	}

	// members are public to make the test flow convenient and keep the unit test simple - but the threads should not use the data directly - they are only allowed to have copies of the shared RCP through getRCPPtr
	RCP<int> commonRCPPtr;												// public so we can error check at the end straight on the ptr from main - we don't want to complicate things by copying the ptr out and then accounting for that ptr's new counters
	static RCPFactory rcpFactory;										// the singleton factory class
	std::mutex mutexForPtrAccess;										// used to protect getRCPPtr()
	int countNewAllocations = 0;										// this counts each time we call new to make a new RCP ptr data allocation
	int countTotalAccessCalls = 0;										// this tracks all calls to getRCPPtr()
};
RCPFactory RCPFactory::rcpFactory;

// the thread function constantly requests an RCP<int> from the singleton class and then deletes it.
static void get_rcp_from_singleton_many_times(int totalAllocations) {
	for(int n = 0; n < totalAllocations; ++n ) {
		RCP<int> rcpPtr = RCPFactory::rcpFactory.getRCPPtr(); // this will constantly allocate and potentially release - move the declaration of rcpPtr out of the loop and it's a much different story.
	}
}

// In this test we have a singleton factory which shares an RCP pointer to may threads
// The key is that the singleton will dump memory when all access goes away
// So the counter will constantly go to 0 (dump memory) or from 0 to 1 (allocate with a fresh new)
// The initial failures identified by this test were intermittent so required a fairly high repeat count
TEUCHOS_UNIT_TEST( RCP, mtRCPFromSingleton )
{
	std::cout << std::endl; // cosmetic

	const int rcpAllocationsPerThread = 100000;
	// run multiple threads, each of which will interact with the singleton and make many copies of the RCP, which then get removed
	std::vector<std::thread> threads;
	for (int i = 0; i < 4; ++i) {
	  threads.push_back(std::thread(get_rcp_from_singleton_many_times, rcpAllocationsPerThread));
	}

	// join and complete all threads
	for (int i = 0; i < threads.size(); ++i) {
	  threads[i].join();
	}

	std::cout << "Total Access Calls: " << RCPFactory::rcpFactory.countTotalAccessCalls
			<< "	Total New Allocations: " << RCPFactory::rcpFactory.countNewAllocations
			<< "	Total Times We Got Shared Data: " << RCPFactory::rcpFactory.countTotalAccessCalls - RCPFactory::rcpFactory.countNewAllocations << std::endl;

	// this is a sanity check - if these numbers don't match up we are not doing what we thought
	int expectedTotalCalls = rcpAllocationsPerThread * threads.size();
	TEST_EQUALITY_CONST(RCPFactory::rcpFactory.countTotalAccessCalls, expectedTotalCalls); // this should be 0 showing all data was deallocated

	// validate the singleton rcp has properly dumped data - we should have a weak count of 1 and a strong count of 0
	// keep in mind that if you call getRCPPtr() directly to check you change the counters by doing so!
	TEST_EQUALITY_CONST(RCPFactory::rcpFactory.commonRCPPtr.weak_count(), 1); 	// this should be one because the static still exists
	TEST_EQUALITY_CONST(RCPFactory::rcpFactory.commonRCPPtr.strong_count(), 0); // this should be 0 showing all data was deallocated

	const int minimumDeviationFromTrivialResult = 20; // for Multi-threading we want to establish the test is doing some minimum amount of variance but this will depend on the machine and many conditions
	if( RCPFactory::rcpFactory.countNewAllocations < minimumDeviationFromTrivialResult || RCPFactory::rcpFactory.countNewAllocations > RCPFactory::rcpFactory.countTotalAccessCalls - minimumDeviationFromTrivialResult ) {
		std::cout << "Multi-threading was not really testing this job: We would like to see more oscillation between sharing and new allocations and this did not hit an arbitrary threshold." << std::endl;
	}
	TEST_COMPARE( RCPFactory::rcpFactory.countNewAllocations, >, minimumDeviationFromTrivialResult );	// if are not actually doing new allocations it's not a good test
	TEST_COMPARE( RCPFactory::rcpFactory.countNewAllocations, <, RCPFactory::rcpFactory.countTotalAccessCalls - minimumDeviationFromTrivialResult );	// if are always doing new allocations it's also not a good test!
}

} // namespace
