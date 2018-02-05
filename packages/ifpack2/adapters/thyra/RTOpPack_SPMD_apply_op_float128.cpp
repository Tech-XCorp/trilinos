
#include "RTOp_Config.h"
#if defined(HAVE_RTOP_EXPLICIT_INSTANTIATION) && defined(HAVE_TEUCHOSCORE_QUADMATH)

#include "RTOpPack_SPMD_apply_op_def.hpp"
#include "Teuchos_ExplicitInstantiationHelpers.hpp"

namespace RTOpPack {

  TEUCHOS_MACRO_TEMPLATE_INSTANT_FLOAT128(RTOPPACK_SPMD_APPLY_OP_INSTANT_SCALAR)

} // namespace RTOpPack

#endif // HAVE_RTOP_EXPLICIT_INSTANTIATION && HAVE_TEUCHOSCORE_QUADMATH
