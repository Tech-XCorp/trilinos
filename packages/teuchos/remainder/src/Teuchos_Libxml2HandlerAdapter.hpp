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

#ifndef TEUCHOS_LIBXML2HANDLERADAPTER_H
#define TEUCHOS_LIBXML2HANDLERADAPTER_H

/*! \file Teuchos_Libxml2HandlerAdapter.hpp
    \brief libxml2 adapter for the TreeBuildingXMLHandler
*/

#include "Teuchos_ConfigDefs.hpp"

#ifdef HAVE_TEUCHOS_LIBXML2
#include <libxml/parser.h>

#include "Teuchos_RCP.hpp"
#include "Teuchos_TreeBuildingXMLHandler.hpp"

extern "C"
{
  /** \ingroup libXML2 callback for start of an XML element. */
  void xmlSAX2StartElement(void* context,
                           const xmlChar* name,
                           const xmlChar** attr);

  /** \ingroup libXML2 callback for end of an XML element. */
  void xmlSAX2EndElement(void* context,
                         const xmlChar* name);

  /** \ingroup libXML2 callback for character data. */
  void xmlSAX2Characters(void* context,
                         const xmlChar* s,
                         int len);
};

#endif


#endif
