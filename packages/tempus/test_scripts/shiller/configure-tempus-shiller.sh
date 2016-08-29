rm -rf C* 
if [ -z "$BUILD_TYPE" ]; then BUILD_TYPE=DEBUG; fi
if [ -z "$TEST" ]; then TEST=OFF; fi
if [ -z "$USE_OPENMP" ]; then USE_OPENMP=OFF; fi
if [ "$USE_OPENMP" == "ON" ]; then 
    NODE_TYPE=OPENMP
elif [ "$USE_PTHREADS" == "ON" ]; then 
    NODE_TYPE=THREAD
elif [ "$USE_CUDA" == "ON" ]; then 
    NODE_TYPE=CUDA
else
    NODE_TYPE=SERIAL
fi

echo NODE_TYPE = ${NODE_TYPE}
echo BUILD_TYPE = ${BUILD_TYPE}
echo Using openmp = ${USE_OPENMP}
echo -D Trilinos_ENABLE_OpenMP=${USE_OPENMP} 

INSTALL_DIR=$WORKSPACE/Trilinos/install-for-drekar/$COMPILER-$BUILD_TYPE

#WORKSPACE=$HOME/Trilinos

cmake \
-D Trilinos_EXTRA_REPOSITORIES="tempus,DrekarResearch,DrekarBase" \
-D Drekar_SYSTEM_TESTS_DIRECTORY:FILEPATH="$WORKSPACE/DrekarSystemTests" \
-D Trilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON \
-D Trilinos_ENABLE_INSTALL_CMAKE_CONFIG_FILES:BOOL=ON \
-D Trilinos_ENABLE_EXAMPLES:BOOL=OFF \
-D Trilinos_ENABLE_TESTS:BOOL=${TEST} \
-D BUILD_SHARED_LIBS:BOOL=OFF \
-D Trilinos_ENABLE_DEBUG=OFF \
-D CMAKE_BUILD_TYPE:STRING=${BUILD_TYPE} \
-D Phalanx_KOKKOS_DEVICE_TYPE:STRING="${NODE_TYPE}" \
-D Trilinos_ENABLE_Fortran:BOOL=ON \
-D HAVE_INTREPID_KOKKOSCORE:BOOL=ON \
-D Panzer_ENABLE_FADTYPE:STRING="Sacado::Fad::DFad<RealType>" \
-D Panzer_ENABLE_TESTS:BOOL=ON \
-D Trilinos_ENABLE_KokkosCore:BOOL=ON \
-D Trilinos_ENABLE_KokkosAlgorithms:BOOL=ON \
-D Trilinos_ENABLE_ALL_PACKAGES:BOOL=OFF \
-D Trilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF \
-D Trilinos_ENABLE_Teko:BOOL=ON \
-D Trilinos_ENABLE_Belos:BOOL=ON \
-D Trilinos_ENABLE_Panzer:BOOL=ON \
-D Trilinos_ENABLE_Shards:BOOL=ON \
-D Trilinos_ENABLE_Stratimikos:BOOL=ON \
-D Trilinos_ENABLE_ML:BOOL=ON \
-D Trilinos_ENABLE_Zoltan:BOOL=ON \
-D Trilinos_ENABLE_FEI:BOOL=ON \
-D Trilinos_ENABLE_Amesos:BOOL=ON \
-D Trilinos_ENABLE_SEACAS:BOOL=ON \
-D Trilinos_ENABLE_SEACASIoss:BOOL=ON \
-D Trilinos_ENABLE_STK:BOOL=ON \
-D Trilinos_ENABLE_STKClassic:BOOL=OFF \
-D Trilinos_ENABLE_STKMesh:BOOL=ON \
-D Trilinos_ENABLE_STKUtil:BOOL=ON \
-D Trilinos_ENABLE_STKSearch:BOOL=OFF \
-D Trilinos_ENABLE_STKTopology:BOOL=ON \
-D Trilinos_ENABLE_STKTransfer:BOOL=ON \
-D Trilinos_ENABLE_STKDoc_tests:BOOL=OFF \
-D Trilinos_ENABLE_STKUnit_tests:BOOL=OFF \
-D Trilinos_ENABLE_STKUnit_test_utils:BOOL=OFF \
-D TPL_ENABLE_GLM=OFF \
-D Trilinos_ENABLE_Stokhos:BOOL=OFF \
-D Trilinos_ENABLE_Tempus:BOOL=ON \
-D Trilinos_ENABLE_Drekar:BOOL=ON \
-D Trilinos_ENABLE_DrekarMHD:BOOL=ON \
-D Drekar_ENABLE_TESTS:BOOL=ON \
-D Drekar_ENABLE_EXTENDED_SYSTEM_TESTS:BOOL=ON \
-D Drekar_ENABLE_EXAMPLES:BOOL=ON \
-D Drekar_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON \
-D Panzer_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON \
-D SEACASExodus_ENABLE_MPI:BOOL=ON \
-D EpetraExt_ENABLE_HDF5:BOOL=OFF \
-D Teuchos_ENABLE_LONG_LONG_INT:BOOL=OFF \
-D Intrepid_ENABLE_DEBUG_INF_CHECK=OFF \
-D IntrepidIntrepid2_ENABLE_DEBUG_INF_CHECK:BOOL=OFF \
-D CMAKE_CXX_COMPILER:FILEPATH="mpicxx" \
-D CMAKE_C_COMPILER:FILEPATH="mpicc" \
-D CMAKE_Fortran_COMPILER:FILEPATH="mpif77" \
-D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
-D CMAKE_SKIP_RULE_DEPENDENCY=ON \
-D CMAKE_INSTALL_PREFIX:PATH=${INSTALL_DIR} \
-D Trilinos_VERBOSE_CONFIGURE:BOOL=OFF \
-D Trilinos_ENABLE_STRONG_CXX_COMPILE_WARNINGS=OFF \
-D Trilinos_ENABLE_STRONG_C_COMPILE_WARNINGS=OFF \
-D Trilinos_ENABLE_SHADOW_WARNINGS=OFF \
-D TPL_ENABLE_MPI:BOOL=ON \
-D TPL_ENABLE_Boost:BOOL=ON \
-D TPL_ENABLE_Netcdf:BOOS=ON \
-D Netcdf_INCLUDE_DIRS:FILEPATH="${NETCDF_ROOT}/include" \
-D HDF5_INCLUDE_DIRS:FILEPATH="${HDF5_ROOT}/include" \
-D Netcdf_LIBRARY_DIRS:FILEPATH="${NETCDF_ROOT}/lib" \
-D HDF5_LIBRARY_DIRS:FILEPATH="${HDF5_ROOT}/lib" \
-D TPL_HDF5_LIBRARIES:FILEPATH="-L${HDF5_ROOT}/lib;${HDF5_ROOT}/lib/libhdf5_hl.a;${HDF5_ROOT}/lib/libhdf5.a;-lz;-ldl" \
-D TPL_Netcdf_LIBRARIES="-L${BOOST_ROOT}/lib;-L${NETCDF_ROOT}/lib;-L${NETCDF_ROOT}/lib;-L${PNETCDF_ROOT}/lib;-L${HDF5_ROOT}/lib;${BOOST_ROOT}/lib/libboost_program_options.a;${BOOST_ROOT}/lib/libboost_system.a;${NETCDF_ROOT}/lib/libnetcdf.a;${PNETCDF_ROOT}/lib/libpnetcdf.a;${HDF5_ROOT}/lib/libhdf5_hl.a;${HDF5_ROOT}/lib/libhdf5.a;-lz;-ldl" \
-D TPL_ENABLE_BoostLib=ON \
-D TPL_ENABLE_Netcdf:BOOL=ON \
-D TPL_ENABLE_HDF5:BOOL=ON \
-D TPL_ENABLE_Matio=OFF  \
-D TPL_ENABLE_LAPACK:BOOL=ON \
-D TPL_LAPACK_LIBRARIES:FILEPATH="-L${LAPACK_ROOT};-llapack" \
-D TPL_ENABLE_BLAS:BOOL=ON \
-D TPL_BLAS_LIBRARIES:FILEPATH="-L${BLAS_ROOT};-lblas" \
-D CMAKE_SKIP_RULE_DEPENDENCY=ON \
-D TPL_ENABLE_Matio:BOOL=OFF \
-D TPL_ENABLE_X11:BOOL=OFF \
-D TPL_ENABLE_SuperLU:BOOL=OFF \
-D Trilinos_ENABLE_OpenMP=${USE_OPENMP} \
-D Kokkos_ENABLE_OpenMP:BOOL=${USE_OPENMP} \
-D Kokkos_ENABLE_Pthread:BOOL=${USE_PTHREADS} \
-D TPL_ENABLE_CUDA:BOOL=${USE_CUDA} \
-D TPL_ENABLE_CUSPARSE:BOOL=${USE_CUDA} \
-D Kokkos_ENABLE_Cuda_UVM:BOOL=${USE_CUDA} \
-D Kokkos_ENABLE_Debug_Bounds_Check=ON \
-DTrilinos_VERBOSE_CONFIGURE=OFF \
${WORKSPACE}/Trilinos
