#ifndef BELOS_ETI_4ARGUMENT_TPETRA_HPP
#define BELOS_ETI_4ARGUMENT_TPETRA_HPP

// The macro "BELOS_ETI_GROUP" must be defined prior to including this file.

// TODO Eliminate this file?
# include <TpetraCore_config.h>
# include <TpetraCore_ETIHelperMacros.h>
TPETRA_ETI_MANGLING_TYPEDEFS()
TPETRA_INSTANTIATE_SLGN_NO_ORDINAL_SCALAR(BELOS_ETI_GROUP)

#endif //ifndef BELOS_ETI_4ARGUMENT_TPETRA_HPP
