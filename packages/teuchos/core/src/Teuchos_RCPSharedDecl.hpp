/*
 * Teuchos_RCPSharedDecl.hpp
 *
 *  Created on: May 19, 2016
 *      Author: micheldemessieres
 */

#ifndef TEUCHOS_RCP_SHARED_DECL_HPP
#define TEUCHOS_RCP_SHARED_DECL_HPP

#ifdef REFCOUNTPTR_INLINE_FUNCS
#  define REFCOUNTPTR_INLINE inline
#else
#  define REFCOUNTPTR_INLINE
#endif


#ifdef TEUCHOS_DEBUG
#  define TEUCHOS_REFCOUNTPTR_ASSERT_NONNULL
#endif

namespace Teuchos {

enum ERCPWeakNoDealloc { RCP_WEAK_NO_DEALLOC };
enum ERCPUndefinedWeakNoDealloc { RCP_UNDEFINED_WEAK_NO_DEALLOC };
enum ERCPUndefinedWithDealloc { RCP_UNDEFINED_WITH_DEALLOC };

}

#endif // TEUCHOS_RCP_SHARED_DECL_HPP
