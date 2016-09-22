// @HEADER
// ************************************************************************
//
//        Phalanx: A Partial Differential Equation Field Evaluation 
//       Kernel for Flexible Management of Complex Dependency Chains
//                    Copyright 2008 Sandia Corporation
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
// Questions? Contact Roger Pawlowski (rppawlo@sandia.gov), Sandia
// National Laboratories.
//
// ************************************************************************
// @HEADER


#ifndef PHX_MDFIELD_DYN_RANK_DEF_HPP
#define PHX_MDFIELD_DYN_RANK_DEF_HPP

#include <algorithm>
#include <sstream>
#include <vector>
#include <type_traits>
#include <stdexcept>
#include "Teuchos_Assert.hpp"
#include "Teuchos_TypeNameTraits.hpp"
#include "Phalanx_config.hpp"
#include "Phalanx_Print_Utilities.hpp"

//**********************************************************************
//**********************************************************************
// Runtime (Dynamic Rank) Version
//**********************************************************************
//**********************************************************************

//**********************************************************************
template<typename DataT>
KOKKOS_FORCEINLINE_FUNCTION
PHX::MDField<DataT,void,void,void,void,void,void,void,void>::
MDField(const std::string& name, const Teuchos::RCP<PHX::DataLayout>& t) :
  m_tag(name,t)
#ifdef PHX_DEBUG
  , m_tag_set(true),
  m_data_set(false)
#endif
{ }

//**********************************************************************
template<typename DataT>
KOKKOS_FORCEINLINE_FUNCTION
PHX::MDField<DataT,void,void,void,void,void,void,void,void>::
MDField(const PHX::Tag<DataT>& v) :
  m_tag(v)
#ifdef PHX_DEBUG
  ,m_tag_set(true),
  m_data_set(false)
#endif
{ }

//**********************************************************************
template<typename DataT>
KOKKOS_FORCEINLINE_FUNCTION
PHX::MDField<DataT,void,void,void,void,void,void,void,void>::
MDField() :
  m_tag("???", Teuchos::null)
#ifdef PHX_DEBUG
  ,m_tag_set(false),
  m_data_set(false)
#endif
{ }

//**********************************************************************
template<typename DataT>
KOKKOS_FORCEINLINE_FUNCTION
PHX::MDField<DataT,void,void,void,void,void,void,void,void>::
~MDField()
{ }

//**********************************************************************
template<typename DataT>
inline
const PHX::FieldTag& 
PHX::MDField<DataT,void,void,void,void,void,void,void,void>::
fieldTag() const
{ 
#if defined( PHX_DEBUG) && !defined (__CUDA_ARCH__ )
  TEUCHOS_TEST_FOR_EXCEPTION(!m_tag_set, std::logic_error, m_field_tag_error_msg);
#endif
  return m_tag;
}

//**********************************************************************
// template<typename DataT>
// template<typename iType0, typename iType1, typename iType2, typename iType3,
// 	 typename iType4, typename iType5, typename iType6, typename iType7>
// KOKKOS_FORCEINLINE_FUNCTION
// typename PHX::MDFieldTypeTraits<typename PHX::MDField<DataT,void,void,void,void,void,void,void,void>::array_type>::return_type
// PHX::MDField<DataT,void,void,void,void,void,void,void,void>::
// operator()(iType0 index0, iType1 index1, iType2 index2, 
// 	   iType3 index3, iType4 index4, iType5 index5,
// 	   iType6 index6, iType7 index7) const
// { 
// #if defined( PHX_DEBUG) && !defined (__CUDA_ARCH__ )
//   TEUCHOS_TEST_FOR_EXCEPTION(!m_data_set, std::logic_error, m_field_data_error_msg);
//   TEUCHOS_TEST_FOR_EXCEPTION(m_tag.dataLayout().rank() != 8,std::logic_error,"Error: The MDField \"" << m_tag.name() << "\" is of rank " << m_tag.dataLayout().rank() << " but was accessed with the rank 8 operator().");
// #endif

//   TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Runtime PHX::MDField does not support 8 ranks!  Limited to 7.");
//   return m_field_data(index0,index1,index2,index3,index4,index5,index6,index7);
// }

//**********************************************************************
template<typename DataT>
template<typename iType0, typename iType1, typename iType2, typename iType3,
	 typename iType4, typename iType5, typename iType6>
KOKKOS_FORCEINLINE_FUNCTION
typename PHX::MDFieldTypeTraits<typename PHX::MDField<DataT,void,void,void,void,void,void,void,void>::array_type>::return_type
PHX::MDField<DataT,void,void,void,void,void,void,void,void>::
operator()(iType0 index0, iType1 index1, iType2 index2, 
	   iType3 index3, iType4 index4, iType5 index5,
	   iType6 index6) const
{ 
#if defined( PHX_DEBUG) && !defined (__CUDA_ARCH__ )
  TEUCHOS_TEST_FOR_EXCEPTION(!m_data_set, std::logic_error, m_field_data_error_msg);
  TEUCHOS_TEST_FOR_EXCEPTION(m_tag.dataLayout().rank() != 7,std::logic_error,"Error: The MDField \"" << m_tag.name() << "\" is of rank " << m_tag.dataLayout().rank() << " but was accessed with the rank 7 operator().");
#endif
  return m_field_data(index0,index1,index2,index3,index4,index5,index6);
}

//**********************************************************************
template<typename DataT>
template<typename iType0, typename iType1, typename iType2, typename iType3,
	 typename iType4, typename iType5>
KOKKOS_FORCEINLINE_FUNCTION
typename PHX::MDFieldTypeTraits<typename PHX::MDField<DataT,void,void,void,void,void,void,void,void>::array_type>::return_type
PHX::MDField<DataT,void,void,void,void,void,void,void,void>::
operator()(iType0 index0, iType1 index1, iType2 index2, 
	   iType3 index3, iType4 index4, iType5 index5) const
{ 
#if defined( PHX_DEBUG) && !defined (__CUDA_ARCH__ )
  TEUCHOS_TEST_FOR_EXCEPTION(!m_data_set, std::logic_error, m_field_data_error_msg);
  TEUCHOS_TEST_FOR_EXCEPTION(m_tag.dataLayout().rank() != 6,std::logic_error,"Error: The MDField \"" << m_tag.name() << "\" is of rank " << m_tag.dataLayout().rank() << " but was accessed with the rank 6 operator().");
#endif
  return m_field_data(index0,index1,index2,index3,index4,index5);
}

//**********************************************************************
template<typename DataT>
template<typename iType0, typename iType1, typename iType2, typename iType3,
	 typename iType4>
KOKKOS_FORCEINLINE_FUNCTION
typename PHX::MDFieldTypeTraits<typename PHX::MDField<DataT,void,void,void,void,void,void,void,void>::array_type>::return_type
PHX::MDField<DataT,void,void,void,void,void,void,void,void>::
operator()(iType0 index0, iType1 index1, iType2 index2, 
	   iType3 index3, iType4 index4) const
{ 
#if defined( PHX_DEBUG) && !defined (__CUDA_ARCH__ )
  TEUCHOS_TEST_FOR_EXCEPTION(!m_data_set, std::logic_error, m_field_data_error_msg);
  TEUCHOS_TEST_FOR_EXCEPTION(m_tag.dataLayout().rank() != 5,std::logic_error,"Error: The MDField \"" << m_tag.name() << "\" is of rank " << m_tag.dataLayout().rank() << " but was accessed with the rank 5 operator().");
#endif
  return m_field_data(index0,index1,index2,index3,index4);
}

//**********************************************************************
template<typename DataT>
template<typename iType0, typename iType1, typename iType2, typename iType3>
KOKKOS_FORCEINLINE_FUNCTION
typename PHX::MDFieldTypeTraits<typename PHX::MDField<DataT,void,void,void,void,void,void,void,void>::array_type>::return_type
PHX::MDField<DataT,void,void,void,void,void,void,void,void>::
operator()(iType0 index0, iType1 index1, iType2 index2, 
	   iType3 index3)const
{ 
#if defined( PHX_DEBUG) && !defined (__CUDA_ARCH__ )
  TEUCHOS_TEST_FOR_EXCEPTION(!m_data_set, std::logic_error, m_field_data_error_msg);
  TEUCHOS_TEST_FOR_EXCEPTION(m_tag.dataLayout().rank() != 4,std::logic_error,"Error: The MDField \"" << m_tag.name() << "\" is of rank " << m_tag.dataLayout().rank() << " but was accessed with the rank 4 operator().");
#endif
  return m_field_data(index0,index1,index2,index3);
}

//**********************************************************************
template<typename DataT>
template<typename iType0, typename iType1, typename iType2>
KOKKOS_FORCEINLINE_FUNCTION
typename PHX::MDFieldTypeTraits<typename PHX::MDField<DataT,void,void,void,void,void,void,void,void>::array_type>::return_type
PHX::MDField<DataT,void,void,void,void,void,void,void,void>::
operator()(iType0 index0, iType1 index1, iType2 index2) const
{ 
#if defined( PHX_DEBUG) && !defined (__CUDA_ARCH__ ) 
  TEUCHOS_TEST_FOR_EXCEPTION(!m_data_set, std::logic_error, m_field_data_error_msg);
  TEUCHOS_TEST_FOR_EXCEPTION(m_tag.dataLayout().rank() != 3,std::logic_error,"Error: The MDField \"" << m_tag.name() << "\" is of rank " << m_tag.dataLayout().rank() << " but was accessed with the rank 3 operator().");
#endif
  return m_field_data(index0,index1,index2);
}

//**********************************************************************
template<typename DataT>
template<typename iType0, typename iType1>
KOKKOS_FORCEINLINE_FUNCTION
typename PHX::MDFieldTypeTraits<typename PHX::MDField<DataT,void,void,void,void,void,void,void,void>::array_type>::return_type
PHX::MDField<DataT,void,void,void,void,void,void,void,void>::
operator()(iType0 index0, iType1 index1) const
{ 
#if defined( PHX_DEBUG) && !defined (__CUDA_ARCH__ )
  TEUCHOS_TEST_FOR_EXCEPTION(!m_data_set, std::logic_error, m_field_data_error_msg);
  TEUCHOS_TEST_FOR_EXCEPTION(m_tag.dataLayout().rank() != 2,std::logic_error,"Error: The MDField \"" << m_tag.name() << "\" is of rank " << m_tag.dataLayout().rank() << " but was accessed with the rank 2 operator().");
#endif
  return m_field_data(index0,index1);
}

//**********************************************************************
template<typename DataT>
template<typename iType0>
KOKKOS_FORCEINLINE_FUNCTION
typename PHX::MDFieldTypeTraits<typename PHX::MDField<DataT,void,void,void,void,void,void,void,void>::array_type>::return_type
PHX::MDField<DataT,void,void,void,void,void,void,void,void>::
operator()(iType0 index0) const
{ 
#if defined( PHX_DEBUG) && !defined (__CUDA_ARCH__ )
  TEUCHOS_TEST_FOR_EXCEPTION(!m_data_set, std::logic_error, m_field_data_error_msg);
  TEUCHOS_TEST_FOR_EXCEPTION(m_tag.dataLayout().rank() != 1,std::logic_error,"Error: The MDField \"" << m_tag.name() << "\" is of rank " << m_tag.dataLayout().rank() << " but was accessed with the rank 1 operator().");
#endif
  return m_field_data(index0);
}

//**********************************************************************
template<typename DataT>
template<typename iType0>
KOKKOS_FORCEINLINE_FUNCTION
typename PHX::MDFieldTypeTraits<typename PHX::MDField<DataT,void,void,void,void,void,void,void,void>::array_type>::return_type
PHX::MDField<DataT,void,void,void,void,void,void,void,void>::
operator[](iType0 index0) const
{ 
#if defined( PHX_DEBUG) && !defined (__CUDA_ARCH__ )
  TEUCHOS_TEST_FOR_EXCEPTION(!m_data_set, std::logic_error, m_field_data_error_msg);
#endif
  return m_field_data[index0];
}

//**********************************************************************
template<typename DataT>
KOKKOS_FORCEINLINE_FUNCTION
typename PHX::MDField<DataT,void,void,void,void,void,void,void,void>::size_type 
PHX::MDField<DataT,void,void,void,void,void,void,void,void>::
rank() const
{ 
#if defined( PHX_DEBUG) && !defined (__CUDA_ARCH__ )
  TEUCHOS_TEST_FOR_EXCEPTION(!m_tag_set, std::logic_error, m_field_data_error_msg);
  TEUCHOS_TEST_FOR_EXCEPTION(!m_data_set, std::logic_error, m_field_data_error_msg);
#endif
  return m_field_data.rank();
}

//**********************************************************************
template<typename DataT>
template<typename iType>
KOKKOS_FORCEINLINE_FUNCTION
typename PHX::MDField<DataT,void,void,void,void,void,void,void,void>::size_type 
PHX::MDField<DataT,void,void,void,void,void,void,void,void>::
dimension(const iType& ord) const
{ 
#if defined( PHX_DEBUG) && !defined (__CUDA_ARCH__ )
  TEUCHOS_TEST_FOR_EXCEPTION(!m_tag_set, std::logic_error, m_field_data_error_msg);
  TEUCHOS_TEST_FOR_EXCEPTION(!m_data_set, std::logic_error, m_field_data_error_msg);
#endif
  return m_field_data.extent(ord);
}

//**********************************************************************
template<typename DataT>
template<typename iType>
KOKKOS_FORCEINLINE_FUNCTION
void PHX::MDField<DataT,void,void,void,void,void,void,void,void>::
dimensions(std::vector<iType>& dims)
{ 
#if defined( PHX_DEBUG) && !defined (__CUDA_ARCH__ )
  TEUCHOS_TEST_FOR_EXCEPTION(!m_tag_set, std::logic_error, m_field_data_error_msg);
  TEUCHOS_TEST_FOR_EXCEPTION(!m_data_set, std::logic_error, m_field_data_error_msg);
#endif
  
  dims.resize(m_tag.dataLayout().rank());
  for ( size_type i = 0 ; i <  m_tag.dataLayout().rank(); ++i ) 
    dims[i] = static_cast<iType>(m_tag.dataLayout().dimension(i)); // dangerous
}

//**********************************************************************
template<typename DataT>
KOKKOS_FORCEINLINE_FUNCTION
typename PHX::MDField<DataT,void,void,void,void,void,void,void,void>::size_type 
PHX::MDField<DataT,void,void,void,void,void,void,void,void>::size() const
{ 
#if defined( PHX_DEBUG) && !defined (__CUDA_ARCH__ )
  TEUCHOS_TEST_FOR_EXCEPTION(!m_data_set, std::logic_error, m_field_data_error_msg);
#endif
  return m_field_data.size();
}

//**********************************************************************
template<typename DataT>
void PHX::MDField<DataT,void,void,void,void,void,void,void,void>::
setFieldTag(const PHX::Tag<DataT>& v)
{  
#ifdef PHX_DEBUG
  m_tag_set = true;
#endif
  m_tag = v;
}

//**********************************************************************
template <typename ViewType>
unsigned PHX::getSacadoSize(const ViewType& view)
{
  return Kokkos::dimension_scalar(view);
}

//**********************************************************************
template<typename DataT>
void PHX::MDField<DataT,void,void,void,void,void,void,void,void>::
setFieldData(const PHX::any& a)
{ 
#if defined( PHX_DEBUG) && !defined (__CUDA_ARCH__ )
  TEUCHOS_TEST_FOR_EXCEPTION(!m_tag_set, std::logic_error, m_field_tag_error_msg);
  m_data_set = true;
#endif

  m_any = a;

  using NonConstDataT = typename std::remove_const<DataT>::type;

  try {
    if (m_tag.dataLayout().rank() == 1)
      m_field_data =  PHX::any_cast<Kokkos::View<NonConstDataT*,PHX::Device>>(a);
    else if (m_tag.dataLayout().rank() == 2)
      m_field_data =  PHX::any_cast<Kokkos::View<NonConstDataT**,PHX::Device>>(a);
    else if (m_tag.dataLayout().rank() == 3)
      m_field_data =  PHX::any_cast<Kokkos::View<NonConstDataT***,PHX::Device>>(a);
    else if (m_tag.dataLayout().rank() == 4)
      m_field_data =  PHX::any_cast<Kokkos::View<NonConstDataT****,PHX::Device>>(a);
    else if (m_tag.dataLayout().rank() == 5)
      m_field_data =  PHX::any_cast<Kokkos::View<NonConstDataT*****,PHX::Device>>(a);
    else if (m_tag.dataLayout().rank() == 6)
      m_field_data =  PHX::any_cast<Kokkos::View<NonConstDataT******,PHX::Device>>(a);
    else if (m_tag.dataLayout().rank() == 7)
      m_field_data =  PHX::any_cast<Kokkos::View<NonConstDataT*******,PHX::Device>>(a);
    else {
      throw std::runtime_error("ERROR - PHX::MDField::setFieldData (DynRank) - Invalid rank!");
    }
  }
  catch (std::exception& e) {
    
    //std::string type_cast_name = Teuchos::demangleName(typeid(non_const_view).name());
    std::string type_cast_name = "???";

    std::cout << "\n\nError in runtime PHX::MDField::setFieldData() in PHX::any_cast. Tried to cast the field \"" 
	      << this->fieldTag().name()  << "\" with the identifier \"" << this->fieldTag().identifier() 
	      << "\" to a type of \"" << type_cast_name
	      << "\" from a PHX::any object containing a type of \"" 
	      << Teuchos::demangleName(a.type().name()) << "\"." << std::endl;
    throw;
  }

}

//**********************************************************************
template<typename DataT>
void PHX::MDField<DataT,void,void,void,void,void,void,void,void>::
print(std::ostream& os,	bool printValues) const
{

  os << "MDField(";
  for (size_type i=0; i < m_tag.dataLayout().rank(); ++i) {
    if (i > 0)
      os << ",";
    os << m_tag.dataLayout().dimension(i);
  }
  os << "): ";
  
  os << m_tag;

  if (printValues)
    os << "Error - MDField no longer supports the \"printValues\" member of the MDField::print() method. Values may be on a device that does not support printing (e.g. GPU).  Please disconstinue the use of this call!" << std::endl;  

}
//**********************************************************************
template<typename DataT>
template<typename MDFieldType>
void
PHX::MDField<DataT,void,void,void,void,void,void,void,void>::
deep_copy(const MDFieldType& source)
{
  Kokkos::deep_copy(m_field_data,source.get_view());
}
//******************************************************************************

template<typename DataT>
template<typename MDFieldTypeA, typename MDFieldTypeB, unsigned int RANK>
KOKKOS_INLINE_FUNCTION
void
PHX::MDField<DataT,void,void,void,void,void,void,void,void>::
V_MultiplyFunctor<MDFieldTypeA, MDFieldTypeB, RANK>::operator() (const int & ind1) const
{
  if (RANK == 1){
    base_.m_field_data(ind1) = base_.m_field_data(ind1)*source_(ind1);
  }
  else if (RANK == 2){
    for (int ind2=0; ind2<(int)base_.m_field_data.dimension(1); ind2++)
      base_.m_field_data(ind1,ind2) = base_.m_field_data(ind1,ind2)*source_(ind1,ind2);
  }
   else if (RANK == 3){
     for (int ind2=0; ind2<(int)base_.m_field_data.dimension(1); ind2++)
       for (int ind3=0; ind3<(int)base_.m_field_data.dimension(2); ind3++)
         base_.m_field_data(ind1,ind2,ind3) = base_.m_field_data(ind1,ind2,ind3)*source_(ind1,ind2,ind3);
   }
   else if (RANK == 4){
     for (int ind2=0; ind2<(int)base_.m_field_data.dimension(1); ind2++)
       for (int ind3=0; ind3<(int)base_.m_field_data.dimension(2); ind3++)
         for (int ind4=0; ind4<(int)base_.m_field_data.dimension(3); ind4++)
           base_.m_field_data(ind1,ind2,ind3,ind4) = base_.m_field_data(ind1,ind2,ind3,ind4)*source_(ind1,ind2,ind3,ind4);
   }
   else if (RANK == 5){
     for (int ind2=0; ind2<(int)base_.m_field_data.dimension(1); ind2++)
       for (int ind3=0; ind3<(int)base_.m_field_data.dimension(2); ind3++)
         for (int ind4=0; ind4<(int)base_.m_field_data.dimension(3); ind4++)
           for (int ind5=0; ind5<(int)base_.m_field_data.dimension(4); ind5++)
             base_.m_field_data(ind1,ind2,ind3,ind4,ind5) = base_.m_field_data(ind1,ind2,ind3,ind4,ind5)*source_(ind1,ind2,ind3,ind4,ind5);
   }
   else if (RANK == 6){
     for (int ind2=0; ind2<(int)base_.m_field_data.dimension(1); ind2++)
       for (int ind3=0; ind3<(int)base_.m_field_data.dimension(2); ind3++)
         for (int ind4=0; ind4<(int)base_.m_field_data.dimension(3); ind4++)
           for (int ind5=0; ind5<(int)base_.m_field_data.dimension(4); ind5++)
             for (int ind6=0; ind6<(int)base_.m_field_data.dimension(5); ind6++)
               base_.m_field_data(ind1,ind2,ind3,ind4,ind5,ind6) = base_.m_field_data(ind1,ind2,ind3,ind4,ind5,ind6)*source_(ind1,ind2,ind3,ind4,ind5,ind6);
   }
   else if (RANK == 7){
     for (int ind2=0; ind2<(int)base_.m_field_data.dimension(1); ind2++)
       for (int ind3=0; ind3<(int)base_.m_field_data.dimension(2); ind3++)
         for (int ind4=0; ind4<(int)base_.m_field_data.dimension(3); ind4++)
           for (int ind5=0; ind5<(int)base_.m_field_data.dimension(4); ind5++)
             for (int ind6=0; ind6<(int)base_.m_field_data.dimension(5); ind6++)
               for (int ind7=0; ind7<(int)base_.m_field_data.dimension(6); ind7++)
                 base_.m_field_data(ind1,ind2,ind3,ind4,ind5,ind6,ind7) = base_.m_field_data(ind1,ind2,ind3,ind4,ind5,ind6,ind7)*source_(ind1,ind2,ind3,ind4,ind5,ind6,ind7);
   }
 }

//******************************************************************************
template<typename DataT>
template<typename MDFieldType>
void
PHX::MDField<DataT,void,void,void,void,void,void,void,void>::
V_Multiply(const MDFieldType& source)
{
  typedef PHX::MDField<DataT,void,void,void,void,void,void,void,void> ThisType;
  const auto length = m_tag.dataLayout().dimension(0);
  if (m_tag.dataLayout().rank() == 1){
    Kokkos::parallel_for( length, V_MultiplyFunctor<ThisType, MDFieldType, 1>(*this, source) );
  }
  else if (m_tag.dataLayout().rank() == 2){
    Kokkos::parallel_for( length, V_MultiplyFunctor<ThisType, MDFieldType, 2>(*this, source) );
 }
  else if (m_tag.dataLayout().rank() == 3){
    Kokkos::parallel_for( length, V_MultiplyFunctor<ThisType, MDFieldType, 3>(*this, source) );
  }
  else if (m_tag.dataLayout().rank() == 4){
    Kokkos::parallel_for( length, V_MultiplyFunctor<ThisType, MDFieldType, 4>(*this, source) );
  }
  else if (m_tag.dataLayout().rank() == 5){
    Kokkos::parallel_for( length, V_MultiplyFunctor<ThisType, MDFieldType, 5>(*this, source) );
  }
  else if (m_tag.dataLayout().rank() == 6){
    Kokkos::parallel_for( length, V_MultiplyFunctor<ThisType, MDFieldType, 6>(*this, source) );
  }
  else if (m_tag.dataLayout().rank() == 7){
    Kokkos::parallel_for( length, V_MultiplyFunctor<ThisType, MDFieldType, 7>(*this, source) );
  }
}

//******************************************************************************
template<typename DataT>
void
PHX::MDField<DataT,void,void,void,void,void,void,void,void>::
deep_copy(const DataT source)
{
  Kokkos::deep_copy(m_field_data, source);  
}

//**********************************************************************
template<typename DataT>
std::ostream& PHX::operator<<(std::ostream& os, 
			      const PHX::MDField<DataT,void,void,
			      void,void,void,void,void,void>& f)
{
  f.print(os, false);
  return os;
}

//**********************************************************************
#endif
