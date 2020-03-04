// @HEADER
//
// ***********************************************************************
//
//           Amesos2: Templated Direct Sparse Solver Package
//                  Copyright 2011 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
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
//
// @HEADER

/**
   \file   Amesos2_Superlu_TypeMap.hpp
   \author Eric Bavier <etbavie@sandia.gov>
   \date   Mon May 31 23:12:32 2010

   \brief Provides definition of SuperLU types as well as conversions and type
	  traits.

*/

#ifndef AMESOS2_SUPERLU_TYPEMAP_HPP
#define AMESOS2_SUPERLU_TYPEMAP_HPP

#include <functional>
#ifdef HAVE_TEUCHOS_COMPLEX
#include <complex>
#endif

#include <Teuchos_as.hpp>
#ifdef HAVE_TEUCHOS_COMPLEX
#include <Teuchos_SerializationTraits.hpp>
#endif

#include "Amesos2_TypeMap.hpp"

namespace SLU {

typedef int int_t;

extern "C" {

#undef __SUPERLU_SUPERMATRIX
#include "supermatrix.h"	// for Dtype_t declaration
} // end extern "C"

} // end namespace SLU

namespace Amesos2 {

template <class, class> class Superlu;

/* Specialize the Amesos2::TypeMap struct for Superlu types
 *
 * \cond Superlu_type_specializations
 */
template <>
struct TypeMap<Superlu,float>
{
  static SLU::Dtype_t dtype;
  typedef float type;
  typedef float magnitude_type;
};


template <>
struct TypeMap<Superlu,double>
{
  static SLU::Dtype_t dtype;
  typedef double type;
  typedef double magnitude_type;
};


#ifdef HAVE_TEUCHOS_COMPLEX

template <>
struct TypeMap<Superlu,std::complex<float> >
{
  static SLU::Dtype_t dtype;
  typedef Kokkos::complex<float> type;
  typedef float magnitude_type;
};


template <>
struct TypeMap<Superlu,std::complex<double> >
{
  static SLU::Dtype_t dtype;
  typedef Kokkos::complex<double> type;
  typedef double magnitude_type;
};


template <>
struct TypeMap<Superlu,Kokkos::complex<float> >
{
  static SLU::Dtype_t dtype;
  typedef Kokkos::complex<float> type;
  typedef float magnitude_type;
};


template <>
struct TypeMap<Superlu,Kokkos::complex<double> >
{
  static SLU::Dtype_t dtype;
  typedef Kokkos::complex<double> type;
  typedef double magnitude_type;
};


#endif  // HAVE_TEUCHOS_COMPLEX

/* \endcond Superlu_type_specializations */


} // end namespace Amesos2

#endif  // AMESOS2_SUPERLU_TYPEMAP_HPP
