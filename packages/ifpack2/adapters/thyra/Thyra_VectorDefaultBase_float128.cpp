
#include "Thyra_Config.h"
#if defined(HAVE_THYRA_EXPLICIT_INSTANTIATION) && defined(HAVE_TEUCHOSCORE_QUADMATH)

#include "Teuchos_ExplicitInstantiationHelpers.hpp"
#include "Thyra_VectorDefaultBase_decl.hpp"
#include "Thyra_VectorDefaultBase_def.hpp"

namespace Thyra {

TEUCHOS_CLASS_TEMPLATE_INSTANT_FLOAT128(VectorDefaultBase)

} // namespace Thyra

#endif // HAVE_THYRA_EXPLICIT_INSTANTIATION && HAVE_TEUCHOSCORE_QUADMATH
