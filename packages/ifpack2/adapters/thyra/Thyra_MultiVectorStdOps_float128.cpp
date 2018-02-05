
#include "Thyra_Config.h"
#if defined(HAVE_THYRA_EXPLICIT_INSTANTIATION) && defined(HAVE_TEUCHOSCORE_QUADMATH)

#include "Teuchos_ExplicitInstantiationHelpers.hpp"
#include "Thyra_MultiVectorStdOps_decl.hpp"
#include "Thyra_MultiVectorStdOps_def.hpp"

namespace Thyra {

TEUCHOS_MACRO_TEMPLATE_INSTANT_FLOAT128(THYRA_MULTI_VECTOR_STD_OPS_INSTANT)

} // namespace Thyra

#endif // HAVE_THYRA_EXPLICIT_INSTANTIATION && HAVE_TEUCHOSCORE_QUADMATH
