TRIBITS_PACKAGE_DEFINE_DEPENDENCIES(
  SUBPACKAGES_DIRS_CLASSIFICATIONS_OPTREQS
    Core          core          PS  REQUIRED
    ParameterList parameterlist PS  REQUIRED
    Comm          comm          PS  REQUIRED
    Numerics      numerics      PS  REQUIRED
    Remainder     remainder     PS  REQUIRED
    KokkosCompat  kokkoscompat  PS  OPTIONAL
    KokkosComm    kokkoscomm    PS  OPTIONAL
  )

SET(LIB_OPTIONAL_DEP_TPLS Pthread)

TRIBITS_TPL_TENTATIVELY_ENABLE(Pthread)
