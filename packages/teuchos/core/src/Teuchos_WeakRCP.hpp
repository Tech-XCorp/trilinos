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

#ifndef TEUCHOS_WEAKRCP_HPP
#define TEUCHOS_WEAKRCP_HPP


/*! \file Teuchos_RCP.hpp
    \brief Reference-counted pointer class and non-member templated function implementations.
*/

/** \example example/RefCountPtr/cxx_main.cpp
    This is an example of how to use the <tt>Teuchos::RCP</tt> class.
*/

/** \example test/MemoryManagement/RCP_test.cpp
    This is a more detailed testing program that uses all of the <tt>Teuchos::RCP</tt> class.
*/

#include "Teuchos_RCPDecl.hpp"
#include "Teuchos_WeakRCPDecl.hpp"
#include "Teuchos_Ptr.hpp"
#include "Teuchos_Assert.hpp"
#include "Teuchos_Exceptions.hpp"
#include "Teuchos_dyn_cast.hpp"
#include "Teuchos_map.hpp"
#include "Teuchos_TypeNameTraits.hpp"


namespace Teuchos {


// very bad public functions

template<class T>
inline
WeakRCP<T>::WeakRCP( T* p, const RCPNodeHandle& node)
  : ptr_(p), node_(node)
{}


template<class T>
inline
T* WeakRCP<T>::access_private_ptr() const
{  return ptr_; }


template<class T>
inline
RCPNodeHandle& WeakRCP<T>::nonconst_access_private_node()
{  return node_; }


template<class T>
inline
const RCPNodeHandle& WeakRCP<T>::access_private_node() const
{  return node_; }




// Constructors/destructors/initializers


template<class T>
inline
WeakRCP<T>::WeakRCP( ENull )
  : ptr_(NULL)
{}


template<class T>
inline
WeakRCP<T>::WeakRCP( T* p, ERCPWeakNoDealloc )
  : ptr_(p)
#ifndef TEUCHOS_DEBUG
  , node_(RCP_createNewRCPNodeRawPtrNonowned(p))
#endif // TEUCHOS_DEBUG
{
#ifdef TEUCHOS_DEBUG
  if (p) {
    RCPNode* existing_RCPNode = RCPNodeTracer::getExistingRCPNode(p);
    if (existing_RCPNode) {
      // Will not call add_new_RCPNode(...)
      node_ = RCPNodeHandle(existing_RCPNode, RCP_WEAK, false);
    }
    else {
      // Will call add_new_RCPNode(...)
      node_ = RCPNodeHandle(
        RCP_createNewRCPNodeRawPtrNonowned(p),
        p, typeName(*p), concreteTypeName(*p),
        false
        );
    }
  }
#endif // TEUCHOS_DEBUG
}


template<class T>
inline
WeakRCP<T>::WeakRCP( T* p, ERCPUndefinedWeakNoDealloc )
  : ptr_(p),
    node_(RCP_createNewRCPNodeRawPtrNonownedUndefined(p))
{}


template<class T>
inline
WeakRCP<T>::WeakRCP( T* p, bool has_ownership_in )
  : ptr_(p)
#ifndef TEUCHOS_DEBUG
  , node_(RCP_createNewRCPNodeRawPtr(p, has_ownership_in))
#endif // TEUCHOS_DEBUG
{
#ifdef TEUCHOS_DEBUG
  if (p) {
    RCPNode* existing_RCPNode = 0;
    if (!has_ownership_in) {
      existing_RCPNode = RCPNodeTracer::getExistingRCPNode(p);
    }
    if (existing_RCPNode) {
      // Will not call add_new_RCPNode(...)
      node_ = RCPNodeHandle(existing_RCPNode, RCP_WEAK, false);
    }
    else {
      // Will call add_new_RCPNode(...)
      RCPNodeThrowDeleter nodeDeleter(RCP_createNewRCPNodeRawPtr(p, has_ownership_in));
      node_ = RCPNodeHandle(
        nodeDeleter.get(),
        p, typeName(*p), concreteTypeName(*p),
        has_ownership_in
        );
      nodeDeleter.release();
    }
  }
#endif // TEUCHOS_DEBUG
}

template<class T>
inline
WeakRCP<T>::WeakRCP(const WeakRCP<T>& r_ptr)
  : ptr_(r_ptr.ptr_), node_(r_ptr.node_)
{}


template<class T>
template<class T2>
inline
WeakRCP<T>::WeakRCP(const WeakRCP<T2>& r_ptr)
  : ptr_(r_ptr.access_private_ptr()), // will not compile if T is not base class of T2
    node_(r_ptr.access_private_node())
{}


template<class T>
inline
WeakRCP<T>::~WeakRCP()
{}


template<class T>
inline
WeakRCP<T>& WeakRCP<T>::operator=(const WeakRCP<T>& r_ptr)
{
#ifdef TEUCHOS_DEBUG
  if (this == &r_ptr)
    return *this;
  reset(); // Force delete first in debug mode!
#endif
  WeakRCP<T>(r_ptr).swap(*this);
  return *this;
}


template<class T>
inline
WeakRCP<T>& WeakRCP<T>::operator=(ENull)
{
  reset();
  return *this;
}


template<class T>
inline
void WeakRCP<T>::swap(WeakRCP<T> &r_ptr)
{
  std::swap(r_ptr.ptr_, ptr_);
  node_.swap(r_ptr.node_);
}


// Object query and access functions


template<class T>
inline
bool WeakRCP<T>::is_null() const
{
  return ptr_ == 0;
}

template<class T>
inline
WeakRCP<const T> WeakRCP<T>::getConst() const
{
  return rcp_implicit_cast<const T>(*this);
}


// Reference counting


template<class T>
inline
ERCPStrength WeakRCP<T>::strength() const
{
  return node_.strength();
}


template<class T>
inline
bool WeakRCP<T>::is_valid_ptr() const
{
  if (ptr_)
    return node_.is_valid_ptr();
  return true;
}


template<class T>
inline
int WeakRCP<T>::strong_count() const
{
  return node_.strong_count();
}


template<class T>
inline
int WeakRCP<T>::weak_count() const
{
  return node_.weak_count();
}


template<class T>
inline
int WeakRCP<T>::total_count() const
{
  return node_.total_count();
}


template<class T>
inline
void WeakRCP<T>::set_has_ownership()
{
  node_.has_ownership(true);
}


template<class T>
inline
bool WeakRCP<T>::has_ownership() const
{
  return node_.has_ownership();
}

template<class T>
inline
WeakRCP<T> WeakRCP<T>::create_weak() const
{
  debug_assert_valid_ptr();
  return WeakRCP<T>(ptr_, node_.create_weak());
}


template<class T>
inline
RCP<T> WeakRCP<T>::create_strong() const
{
//  debug_assert_valid_ptr();
  if (is_null()) {
    return RCP<T>(null); // return a null RCP - we are no good
  }

  // try the weak to strong conversion - thread safe and not guaranteed to succeed
  RCPNodeHandle strongHandle = node_.attempt_create_strong_from_possible_weak_only();
  if (strongHandle.strong_count() == 0) {
    return RCP<T>(null); // return a null RCP - we failed
  }
  return RCP<T>(ptr_, strongHandle);
}


template<class T>
template <class T2>
inline
bool WeakRCP<T>::shares_resource(const WeakRCP<T2>& r_ptr) const
{
  return node_.same_node(r_ptr.access_private_node());
  // Note: above, r_ptr is *not* the same class type as *this so we can not
  // access its node_ member directly!  This is an interesting detail to the
  // C++ protected/private protection mechanism!
}

template<class T>
template <class T2>
inline
bool WeakRCP<T>::shares_resource(const RCP<T2>& r_ptr) const
{
  return node_.same_node(r_ptr.access_private_node());
  // Note: above, r_ptr is *not* the same class type as *this so we can not
  // access its node_ member directly!  This is an interesting detail to the
  // C++ protected/private protection mechanism!
}


// Assertions


template<class T>
inline
const WeakRCP<T>& WeakRCP<T>::assert_not_null() const
{
  if (!ptr_)
    throw_null_ptr_error(typeName(*this));
  return *this;
}


template<class T>
inline
const WeakRCP<T>& WeakRCP<T>::assert_valid_ptr() const
{
  if (ptr_)
    node_.assert_valid_ptr(*this);
  return *this;
}


// boost::shared_ptr compatiblity funtions


template<class T>
inline
void WeakRCP<T>::reset()
{
#ifdef TEUCHOS_DEBUG
  node_ = RCPNodeHandle();
#else
  RCPNodeHandle().swap(node_);
#endif
  ptr_ = 0;
}


template<class T>
template<class T2>
inline
void WeakRCP<T>::reset(T2* p, bool has_ownership_in)
{
  *this = rcp(p, has_ownership_in);
}


template<class T>
inline
int WeakRCP<T>::count() const
{
  return node_.count();
}

}  // end namespace Teuchos



template<class T>
inline
bool Teuchos::is_null( const WeakRCP<T> &p )
{
  return p.is_null();
}


template<class T>
inline
bool Teuchos::nonnull( const WeakRCP<T> &p )
{
  return !p.is_null();
}


template<class T>
inline
bool Teuchos::operator==( const WeakRCP<T> &p, ENull )
{
  return p.access_private_ptr() == NULL;
}


template<class T>
inline
bool Teuchos::operator!=( const WeakRCP<T> &p, ENull )
{
  return p.access_private_ptr() != NULL;
}


template<class T1, class T2>
inline
bool Teuchos::operator==( const WeakRCP<T1> &p1, const WeakRCP<T2> &p2 )
{
  return p1.access_private_node().same_node(p2.access_private_node());
}


template<class T1, class T2>
inline
bool Teuchos::operator!=( const WeakRCP<T1> &p1, const WeakRCP<T2> &p2 )
{
  return !p1.access_private_node().same_node(p2.access_private_node());
}

template<class T>
std::ostream& Teuchos::operator<<( std::ostream& out, const WeakRCP<T>& p )
{
  out
    << typeName(p) << "{"
    << "ptr="<<(const void*)(p.get()) // I can't find any alternative to this C cast :-(
    <<",node="<<p.access_private_node()
    <<",strong_count="<<p.strong_count()
    <<",weak_count="<<p.weak_count()
    <<"}";
  return out;
}


#endif // TEUCHOS_WEAKRCP_HPP
