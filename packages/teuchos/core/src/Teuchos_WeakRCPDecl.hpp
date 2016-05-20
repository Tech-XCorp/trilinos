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

#ifndef TEUCHOS_WEAKRCP_DECL_HPP
#define TEUCHOS_WEAKRCP_DECL_HPP

/*! \file Teuchos_WeakRCPDecl.hpp
    \brief Reference-counted pointer class and non-member templated function implementations.
*/

#include "Teuchos_RCPSharedDecl.hpp"
#include "Teuchos_RCPDecl.hpp"
#include "Teuchos_RCPNode.hpp"
#include "Teuchos_ENull.hpp"
#include "Teuchos_NullIteratorTraits.hpp"

namespace Teuchos {

/** \brief . */
template<class T> class Ptr;
template<class T> class RCP;

template<class T>
class WeakRCP {
public:

  /** \brief . */
  typedef T  element_type;

  /** \name Constructors/destructors/initializers. */
  //@{

  /** \brief Initialize <tt>RCP<T></tt> to NULL.
   *
   * <b>Postconditons:</b> <ul>
   * <li> <tt>this->get() == 0</tt>
   * <li> <tt>this->strength() == RCP_STRENGTH_INVALID</tt>
   * <li> <tt>this->is_vali_ptr() == true</tt>
   * <li> <tt>this->strong_count() == 0</tt>
   * <li> <tt>this->weak_count() == 0</tt>
   * <li> <tt>this->has_ownership() == false</tt>
   * </ul>
   *
   * This allows clients to write code like:
   \code
   RCP<int> p = null;
   \endcode
   or
   \code
   RCP<int> p;
   \endcode
   * and construct to <tt>NULL</tt>
   */
  inline WeakRCP(ENull null_arg = null);

  /** \brief Construct from a raw pointer.
   *
   * Note that this constructor is declared explicit so there is no implicit
   * conversion from a raw pointer to an RCP allowed.  If
   * <tt>has_ownership==false</tt>, then no attempt to delete the object will
   * occur.
   *
   * <b>Postconditons:</b><ul>
   * <li> <tt>this->get() == p</tt>
   * <li> <tt>this->strength() == RCP_STRONG</tt>
   * <li> <tt>this->is_vali_ptr() == true</tt>
   * <li> <tt>this->strong_count() == 1</tt>
   * <li> <tt>this->weak_count() == 0</tt>
   * <li> <tt>this->has_ownership() == has_ownership</tt>
   * </ul>
   *
   * NOTE: It is recommended that this constructor never be called directly
   * but only through a type-specific non-member constructor function or at
   * least through the general non-member <tt>rcp()</tt> function.
   */
  inline explicit WeakRCP( T* p, bool has_ownership = true );

  /** \brief Initialize from another <tt>RCP<T></tt> object.
   *
   * After construction, <tt>this</tt> and <tt>r_ptr</tt> will
   * reference the same object.
   *
   * This form of the copy constructor is required even though the
   * below more general templated version is sufficient since some
   * compilers will generate this function automatically which will
   * give an incorrect implementation.
   *
   * <b>Postconditons:</b><ul>
   * <li> <tt>this->get() == r_ptr.get()</tt>
   * <li> <tt>this->strong_count() == r_ptr.strong_count()</tt>
   * <li> <tt>this->has_ownership() == r_ptr.has_ownership()</tt>
   * <li> If <tt>r_ptr.get() != NULL</tt> then <tt>r_ptr.strong_count()</tt> is incremented by 1
   * </ul>
   */
  inline WeakRCP(const WeakRCP<T>& r_ptr);

  /** \brief Initialize from another <tt>RCP<T2></tt> object (implicit conversion only).
   *
   * This function allows the implicit conversion of smart pointer objects just
   * like with raw C++ pointers.  Note that this function will only compile
   * if the statement <tt>T1 *ptr = r_ptr.get()</tt> will compile.
   *
   * <b>Postconditons:</b> <ul>
   * <li> <tt>this->get() == r_ptr.get()</tt>
   * <li> <tt>this->strong_count() == r_ptr.strong_count()</tt>
   * <li> <tt>this->has_ownership() == r_ptr.has_ownership()</tt>
   * <li> If <tt>r_ptr.get() != NULL</tt> then <tt>r_ptr.strong_count()</tt> is incremented by 1
   * </ul>
   */
  template<class T2>
  inline WeakRCP(const WeakRCP<T2>& r_ptr);

  /** \brief Removes a reference to a dynamically allocated object and possibly deletes
   * the object if owned.
   *
   * Deletes the object if <tt>this->has_ownership() == true</tt> and
   * <tt>this->strong_count() == 1</tt>.  If <tt>this->strong_count() ==
   * 1</tt> but <tt>this->has_ownership() == false</tt> then the object is not
   * deleted.  If <tt>this->strong_count() > 1</tt> then the internal
   * reference count shared by all the other related <tt>RCP<...></tt> objects
   * for this shared object is deincremented by one.  If <tt>this->get() ==
   * NULL</tt> then nothing happens.
   */
  inline ~WeakRCP();

  /** \brief Copy the pointer to the referenced object and increment the
   * reference count.
   *
   * If <tt>this->has_ownership() == true</tt> and <tt>this->strong_count() == 1</tt>
   * before this operation is called, then the object pointed to by
   * <tt>this->get()</tt> will be deleted (usually using <tt>delete</tt>)
   * prior to binding to the pointer (possibly <tt>NULL</tt>) pointed to in
   * <tt>r_ptr</tt>.  Assignment to self (i.e. <tt>this->get() ==
   * r_ptr.get()</tt>) is harmless and this function does nothing.
   *
   * <b>Postconditons:</b><ul>
   * <li> <tt>this->get() == r_ptr.get()</tt>
   * <li> <tt>this->strong_count() == r_ptr.strong_count()</tt>
   * <li> <tt>this->has_ownership() == r_ptr.has_ownership()</tt>
   * <li> If <tt>r_ptr.get() != NULL</tt> then <tt>r_ptr.strong_count()</tt> is incremented by 1
   * </ul>
   *
   * Provides the "strong guarantee" in a debug build!
   */
  inline WeakRCP<T>& operator=(const WeakRCP<T>& r_ptr);

  /** \brief Assign to null.
   *
   * If <tt>this->has_ownership() == true</tt> and <tt>this->strong_count() == 1</tt>
   * before this operation is called, then the object pointed to by
   * <tt>this->get()</tt> will be deleted (usually using <tt>delete</tt>)
   * prior to binding to the pointer (possibly <tt>NULL</tt>) pointed to in
   * <tt>r_ptr</tt>.
   *
   * <b>Postconditons:</b><ul>
   * <li> See <tt>RCP(ENull)</tt>
   * </ul>
   */
  inline WeakRCP<T>& operator=(ENull);

  /** \brief Swap the contents with some other RCP object. */
  inline void swap(WeakRCP<T> &r_ptr);

  //@}

  /** \name Object/Pointer Access Functions */
  //@{

  /** \brief Returns true if the underlying pointer is null. */
  inline bool is_null() const;

  /** \brief Return an RCP<const T> version of *this. */
  inline WeakRCP<const T> getConst() const;

  //@}

  /** \name Reference counting */
  //@{

  /** \brief Strength of the pointer.
   *
   * Return values:<ul>
   * <li><tt>RCP_STRONG</tt>: Underlying reference-counted object will be deleted
   *     when <tt>*this</tt> is destroyed if <tt>strong_count()==1</tt>.
   * <li><tt>RCP_WEAK</tt>: Underlying reference-counted object will not be deleted
   *     when <tt>*this</tt> is destroyed if <tt>strong_count() > 0</tt>.
   * <li><tt>RCP_STRENGTH_INVALID</tt>: <tt>*this</tt> is not strong or weak but
   *     is null.
   * </ul>
   */
  inline ERCPStrength strength() const;

  /** \brief Return if the underlying object pointer is still valid or not.
   *
   * The underlying object will not be valid if the strong count has gone to
   * zero but the weak count thas not.
   *
   * NOTE: Null is a valid object pointer.  If you want to know if there is a
   * non-null object and it is valid then <tt>!is_null() &&
   * is_valid_ptr()</tt> will be <tt>true</tt>.
   */
  inline bool is_valid_ptr() const;

  /** \brief Return the number of active <tt>RCP<></tt> objects that have a
   * "strong" reference to the underlying reference-counted object.
   *
   * \return If <tt>this->get() == NULL</tt> then this function returns 0.
   */
  inline int strong_count() const;

  /** \brief Return the number of active <tt>RCP<></tt> objects that have a
   * "weak" reference to the underlying reference-counted object.
   *
   * \return If <tt>this->get() == NULL</tt> then this function returns 0.
   */
  inline int weak_count() const;

  /** \brief Total count (strong_count() + weak_count()). */
  inline int total_count() const;

  /** \brief Give <tt>this</tt> and other <tt>RCP<></tt> objects ownership
   * of the referenced object <tt>this->get()</tt>.
   *
   * See ~RCP() above.  This function
   * does nothing if <tt>this->get() == NULL</tt>.
   *
   * <b>Postconditions:</b>
   * <ul>
   * <li> If <tt>this->get() == NULL</tt> then
   *   <ul>
   *   <li> <tt>this->has_ownership() == false</tt> (always!).
   *   </ul>
   * <li> else
   *   <ul>
   *   <li> <tt>this->has_ownership() == true</tt>
   *   </ul>
   * </ul>
   */
  inline void set_has_ownership();

  /** \brief Returns true if <tt>this</tt> has ownership of object pointed to
   * by <tt>this->get()</tt> in order to delete it.
   *
   * See ~RCP() above.
   *
   * \return If this->get() <tt>== NULL</tt> then this function always returns
   * <tt>false</tt>.  Otherwise the value returned from this function depends
   * on which function was called most recently, if any; set_has_ownership()
   * (<tt>true</tt>) or release() (<tt>false</tt>).
   */
  inline bool has_ownership() const;

  /** \brief Create a new weak RCP object from another (strong) RCP object.
   *
   * ToDo: Explain this!
   *
   * <b>Preconditons:</b> <ul>
   * <li> <tt>returnVal.is_valid_ptr()==true</tt>
   * </ul>
   *
   * <b>Postconditons:</b> <ul>
   * <li> <tt>returnVal.get() == this->get()</tt>
   * <li> <tt>returnVal.strong_count() == this->strong_count()</tt>
   * <li> <tt>returnVal.weak_count() == this->weak_count()+1</tt>
   * <li> <tt>returnVal.strength() == RCP_WEAK</tt>
   * <li> <tt>returnVal.has_ownership() == this->has_ownership()</tt>
   * </ul>
   */
  inline WeakRCP<T> create_weak() const;

  /** \brief Create a new strong RCP object from another (weak) RCP object.
   *
   * ToDo: Explain this!
   *
   * <b>Preconditons:</b> <ul>
   * <li> <tt>returnVal.is_valid_ptr()==true</tt>
   * </ul>
   *
   * <b>Postconditons:</b> <ul>
   * <li> <tt>returnVal.get() == this->get()</tt>
   * <li> <tt>returnVal.strong_count() == this->strong_count() + 1</tt>
   * <li> <tt>returnVal.weak_count() == this->weak_count()</tt>
   * <li> <tt>returnVal.strength() == RCP_STRONG</tt>
   * <li> <tt>returnVal.has_ownership() == this->has_ownership()</tt>
   * </ul>
   */
  inline RCP<T> create_strong() const;

  /** \brief Returns true if the smart pointers share the same underlying
   * reference-counted object.
   *
   * This method does more than just check if <tt>this->get() == r_ptr.get()</tt>.
   * It also checks to see if the underlying reference counting machinary is the
   * same.
   */
  template<class T2>
  inline bool shares_resource(const WeakRCP<T2>& r_ptr) const;
  template<class T2>
  inline bool shares_resource(const RCP<T2>& r_ptr) const;

  //@}

  /** \name Assertions */
  //@{

  /** \brief Throws <tt>NullReferenceError</tt> if <tt>this->get()==NULL</tt>,
   * otherwise returns reference to <tt>*this</tt>.
   */
  inline const WeakRCP<T>& assert_not_null() const;

  /** \brief If the object pointer is non-null, assert that it is still valid.
   *
   * If <tt>is_null()==false && strong_count()==0</tt>, this will throw
   * <tt>DanglingReferenceErorr</tt> with a great error message.
   *
   * If <tt>is_null()==true</tt>, then this will not throw any exception.
   *
   * In this context, null is a valid object.
   */
  inline const WeakRCP<T>& assert_valid_ptr() const;

  /** \brief Calls <tt>assert_not_null()</tt> in a debug build. */
  inline const WeakRCP<T>& debug_assert_not_null() const
    {
#ifdef TEUCHOS_REFCOUNTPTR_ASSERT_NONNULL
      assert_not_null();
#endif
      return *this;
    }

  /** \brief Calls <tt>assert_valid_ptr()</tt> in a debug build. */
  inline const WeakRCP<T>& debug_assert_valid_ptr() const
    {
#ifdef TEUCHOS_DEBUG
      assert_valid_ptr();
#endif
      return *this;
    }

  //@}

  /** \name boost::shared_ptr compatiblity funtions. */
  //@{

  /** \brief Reset to null. */
  inline void reset();

  /** \brief Reset the raw pointer with default ownership to delete.
   *
   * Equivalent to calling:

   \code

     r_rcp = rcp(p)

   \endcode
   */
  template<class T2>
  inline void reset(T2* p, bool has_ownership = true);

  /** \brief Returns <tt>strong_count()</tt> [deprecated]. */
  TEUCHOS_DEPRECATED inline int count() const;

  //@}

private:

  // //////////////////////////////////////////////////////////////
  // Private data members

  T *ptr_; // NULL if this pointer is null
  RCPNodeHandle node_; // NULL if this pointer is null

public: // Bad bad bad

  // These constructors are put here because we don't want to confuse users
  // who would otherwise see them.

  /** \brief Construct a non-owning RCP from a raw pointer to a type that *is*
   * defined.
   *
   * This version avoids adding a deallocator but still requires the type to
   * be defined since it looks up the base object's address when doing RCPNode
   * tracing.
   *
   * NOTE: It is recommended that this constructor never be called directly
   * but only through a type-specific non-member constructor function or at
   * least through the general non-member <tt>rcpFromRef()</tt> function.
   */
  inline explicit WeakRCP(T* p, ERCPWeakNoDealloc);

  /** \brief Construct a non-owning RCP from a raw pointer to a type that is
   * *not* defined.
   *
   * This version avoids any type of compile-time queries of the type that
   * would fail due to the type being undefined.
   *
   * NOTE: It is recommended that this constructor never be called directly
   * but only through a type-specific non-member constructor function or at
   * least through the general non-member <tt>rcpFromUndefRef()</tt> function.
   */
  inline explicit WeakRCP(T* p, ERCPUndefinedWeakNoDealloc);

#ifndef DOXYGEN_COMPILE

  // WARNING: A general user should *never* call these functions!
  inline WeakRCP(T* p, const RCPNodeHandle &node);
  inline T* access_private_ptr() const; // Does not throw
  inline RCPNodeHandle& nonconst_access_private_node(); // Does not throw
  inline const RCPNodeHandle& access_private_node() const; // Does not throw

#endif

};

/** \brief Struct for comparing two WeakRCPs. Simply compares
* the raw pointers contained within the RCPs*/
struct WeakRCPComp {
  /** \brief . */
  template<class T1, class T2> inline
  bool operator() (const WeakRCP<T1> p1, const WeakRCP<T2> p2) const{
    return p1.get() < p2.get();
  }
};

/** \brief Struct for comparing two WeakRCPs. Simply compares
* the raw pointers contained within the RCPs*/
struct WeakRCPConstComp {
  /** \brief . */
  template<class T1, class T2> inline
  bool operator() (const WeakRCP<const T1> p1, const WeakRCP<const T2> p2) const{
    return p1.get() < p2.get();
  }
};



// 2008/09/22: rabartl: NOTE: I removed the TypeNameTraits<RCP<T> >
// specialization since I want to be able to print the type name of an RCP
// that does not have the type T fully defined!


/** \brief Traits specialization for RCP.
 *
 * \ingroup teuchos_mem_mng_grp
 */
template<typename T>
class NullIteratorTraits<WeakRCP<T> > {
public:
  static WeakRCP<T> getNull() { return null; }
};

/** \brief Returns true if <tt>p.get()==NULL</tt>.
 *
 * \relates RCP
 */
template<class T> inline
bool is_null( const WeakRCP<T> &p );


/** \brief Returns true if <tt>p.get()!=NULL</tt>.
 *
 * \relates RCP
 */
template<class T> inline
bool nonnull( const WeakRCP<T> &p );


/** \brief Returns true if <tt>p.get()==NULL</tt>.
 *
 * \relates RCP
 */
template<class T> inline
bool operator==( const WeakRCP<T> &p, ENull );


/** \brief Returns true if <tt>p.get()!=NULL</tt>.
 *
 * \relates RCP
 */
template<class T> inline
bool operator!=( const WeakRCP<T> &p, ENull );


/** \brief Return true if two <tt>RCP</tt> objects point to the same
 * referenced-counted object and have the same node.
 *
 * \relates RCP
 */
template<class T1, class T2> inline
bool operator==( const WeakRCP<T1> &p1, const WeakRCP<T2> &p2 );


/** \brief Return true if two <tt>RCP</tt> objects do not point to the
 * same referenced-counted object and have the same node.
 *
 * \relates RCP
 */
template<class T1, class T2> inline
bool operator!=( const WeakRCP<T1> &p1, const WeakRCP<T2> &p2 );


/** \brief Implicit cast of underlying <tt>RCP</tt> type from <tt>T1*</tt> to <tt>T2*</tt>.
 *
 * The function will compile only if (<tt>T2* p2 = p1.get();</tt>) compiles.
 *
 * This is to be used for conversions up an inheritance hierarchy and from non-const to
 * const and any other standard implicit pointer conversions allowed by C++.
 *
 * \relates RCP
 */
template<class T2, class T1> inline
WeakRCP<T2> rcp_implicit_cast(const WeakRCP<T1>& p1);


/** \brief Static cast of underlying <tt>RCP</tt> type from <tt>T1*</tt> to <tt>T2*</tt>.
 *
 * The function will compile only if (<tt>static_cast<T2*>(p1.get());</tt>) compiles.
 *
 * This can safely be used for conversion down an inheritance hierarchy
 * with polymorphic types only if <tt>dynamic_cast<T2>(p1.get()) == static_cast<T2>(p1.get())</tt>.
 * If not then you have to use <tt>rcp_dynamic_cast<tt><T2>(p1)</tt>.
 *
 * \relates RCP
 */
template<class T2, class T1> inline
WeakRCP<T2> rcp_static_cast(const WeakRCP<T1>& p1);


/** \brief Constant cast of underlying <tt>RCP</tt> type from <tt>T1*</tt> to <tt>T2*</tt>.
 *
 * This function will compile only if (<tt>const_cast<T2*>(p1.get());</tt>) compiles.
 *
 * \relates RCP
 */
template<class T2, class T1> inline
WeakRCP<T2> rcp_const_cast(const WeakRCP<T1>& p1);


/** \brief Dynamic cast of underlying <tt>RCP</tt> type from <tt>T1*</tt> to <tt>T2*</tt>.
 *
 * \param p1 [in] The smart pointer casting from
 *
 * \param throw_on_fail [in] If <tt>true</tt> then if the cast fails (for
 * <tt>p1.get()!=NULL) then a <tt>std::bad_cast</tt> std::exception is thrown
 * with a very informative error message.
 *
 * <b>Postconditions:</b><ul>
 * <li> If <tt>( p1.get()!=NULL && throw_on_fail==true && dynamic_cast<T2*>(p1.get())==NULL ) == true</tt>
 *      then an <tt>std::bad_cast</tt> std::exception is thrown with a very informative error message.
 * <li> If <tt>( p1.get()!=NULL && dynamic_cast<T2*>(p1.get())!=NULL ) == true</tt>
 *      then <tt>return.get() == dynamic_cast<T2*>(p1.get())</tt>.
 * <li> If <tt>( p1.get()!=NULL && throw_on_fail==false && dynamic_cast<T2*>(p1.get())==NULL ) == true</tt>
 *      then <tt>return.get() == NULL</tt>.
 * <li> If <tt>( p1.get()==NULL ) == true</tt>
 *      then <tt>return.get() == NULL</tt>.
 * </ul>
 *
 * This function will compile only if (<tt>dynamic_cast<T2*>(p1.get());</tt>) compiles.
 *
 * \relates RCP
 */
template<class T2, class T1> inline
WeakRCP<T2> rcp_dynamic_cast(
  const WeakRCP<T1>& p1, bool throw_on_fail = false
  );

/** \brief Output stream inserter.
 *
 * The implementation of this function just print pointer addresses and
 * therefore puts no restrictions on the data types involved.
 *
 * \relates RCP
 */
template<class T>
std::ostream& operator<<( std::ostream& out, const WeakRCP<T>& p );


} // end namespace Teuchos

#endif  // TEUCHOS_WEAKRCP_DECL_HPP
