#!/usr/bin/env bash
get_cmake_master(){
    CMAKE_COMMIT=09dd52c9d2684e933a3e013abc4f6848cb1befbf
    if [[ -f "cmake-$CMAKE_COMMIT.zip" ]]; then
	echo "using existing"  "cmake-$CMAKE_COMMIT.zip"
    else
	curl -L https://gitlab.kitware.com/cmake/cmake/-/archive/$CMAKE_COMMIT.zip -o cmake-$CMAKE_COMMIT.zip
    fi
    unzip -n -q cmake-$CMAKE_COMMIT.zip
    mkdir -p  cmake-$CMAKE_COMMIT/build
    cd cmake-$CMAKE_COMMIT/build
    if [[ -x "$(command -v cmake)" ]]; then
        cmake -DBUILD_CursesDialog=OFF -DBUILD_TESTING=OFF -DBUILD_QtDialog=OFF -DCMAKE_INSTALL_PREFIX=`pwd`/.. ../
    else
	../bootstrap --parallel=4 --prefix=`pwd`/..
    fi
    make -j4
    make -j4 install
    CMAKE=`pwd`/../bin/cmake
    ${CMAKE} -version
    cd ../..
    return 0
}
get_cmake38(){
	UNAME_S=$(uname -s)
	if [[ ${UNAME_S} == "Linux" ]] || [[ ${UNAME_S} == "Darwin" ]] && [[ $(uname -m) == "x86_64" ]] ; then
	    CMAKE_VER=3.16.8
	    rm -f cmake-${CMAKE_VER}-${UNAME_S}-x86_64.tar.gz
	    curl -L https://github.com/Kitware/CMake/releases/download/v${CMAKE_VER}/cmake-${CMAKE_VER}-${UNAME_S}-x86_64.tar.gz -o cmake-${CMAKE_VER}-${UNAME_S}-x86_64.tar.gz
	    tar xzf cmake-${CMAKE_VER}-${UNAME_S}-x86_64.tar.gz
	    if [[ ${UNAME_S} == "Darwin" ]] ;then
		CMAKE=`pwd`/cmake-${CMAKE_VER}-${UNAME_S}-x86_64/CMake.app/Contents/bin/cmake
	    else
		CMAKE=`pwd`/cmake-${CMAKE_VER}-${UNAME_S}-x86_64/bin/cmake
	    fi
	    return 0
	else
	    return 1
	fi

}
if [[ "$FC" = "ftn"  ]] ; then
    MPIF90="ftn"
    MPICC="cc"
else
    if ! [ -x "$(command -v mpif90)" ]; then
	echo
	echo mpif90 not installed
	echo mpif90 is required for building Scalapack
	echo
	exit 1
    else
	MPIF90="mpif90"
        MPICC=mpicc
    fi
fi
if [[  -z "${FC}" ]]; then
    FC=$($MPIF90 -show|cut -d " " -f 1)
fi
if [[  -z "${NWCHEM_TOP}" ]]; then
    dir3=$(dirname `pwd`)
    dir2=$(dirname "$dir3")
    NWCHEM_TOP=$(dirname "$dir2")
fi
if [[ "$FC" = "ftn"  ]] || [[ ! -z "$USE_CMAKE_MASTER" ]] ; then
    get_cmake_master
else
if [[ -z "${CMAKE}" ]]; then
    #look for cmake
    if [[ -z "$(command -v cmake)" ]]; then
	get_cmake38
	status=$?
	if [ $status -ne 0 ]; then
	    echo cmake required to build scalapack
	    echo Please install cmake
	    echo define the CMAKE env. variable
	    exit 1
	fi
    else
	CMAKE=cmake
    fi
fi
fi
CMAKE_VER_MAJ=$(${CMAKE} --version|cut -d " " -f 3|head -1|cut -d. -f1)
CMAKE_VER_MIN=$(${CMAKE} --version|cut -d " " -f 3|head -1|cut -d. -f2)
echo CMAKE_VER is ${CMAKE_VER_MAJ} ${CMAKE_VER_MIN}
if ((CMAKE_VER_MAJ < 3)) || (((CMAKE_VER_MAJ > 2) && (CMAKE_VER_MIN < 8))); then
    get_cmake38
    status=$?
    if [ $status -ne 0 ]; then
	echo cmake required to build scalapack
	echo Please install cmake
	echo define the CMAKE env. variable
	exit 1
    fi
fi
pwd

#if [[ "$SCALAPACK_SIZE" != "4"  ]] ; then
#    echo SCALAPACK_SIZE must be equal to 4
#    exit 1
#fi
#if [[ "$BLAS_SIZE" != "4"  ]] ; then
#    echo BLAS_SIZE must be equal to 4 for SCALAPACK
#    exit 1
#fi
if [[ "$BLAS_SIZE" != "$SCALAPACK_SIZE"  ]] ; then
    echo "BLAS_SIZE must be the same as SCALAPACK_SIZE"
    echo "BLAS_SIZE = " "$BLAS_SIZE"
    echo "SCALAPACK_SIZE = " "$SCALAPACK_SIZE"
    exit 1
fi

if [[  -z "${SCALAPACK_SIZE}" ]]; then
   SCALAPACK_SIZE=8
fi
if [[ "$BLAS_SIZE" == 4 ]] && [[ -z "$USE_64TO32"   ]] ; then
    if [[ "$NWCHEM_TARGET" != "LINUX" ]] && [[ "$NWCHEM_TARGET" != "MACX" ]] ; then
    echo USE_64TO32 must be set when BLAS_SIZE=4 on 64-bit architectures
    exit 1
    fi
fi
if [[ ! -z "$BUILD_OPENBLAS"   ]] ; then
    BLASOPT="-L`pwd`/../lib -lnwc_openblas"
fi
#git clone https://github.com/scibuilder/scalapack.git
#svn co --non-interactive --trust-server-cert https://icl.utk.edu/svn/scalapack-dev/scalapack/trunk/ scalapack
VERSION=2.1.0
#curl -L https://github.com/Reference-ScaLAPACK/scalapack/archive/v${VERSION}.tar.gz -o scalapack.tgz
#COMMIT=bc6cad585362aa58e05186bb85d4b619080c45a9
COMMIT=ea5d20668a6b8bbee645b7ffe44623c623969d33
rm -rf scalapack 
if [[ -f "scalapack-$COMMIT.zip" ]]; then
    echo "using existing"  "scalapack-$COMMIT.zip"
else
    echo "downloading"  "scalapack-$COMMIT.zip"
    rm -f scalapack-$COMMIT.zip
    curl -L https://github.com/Reference-ScaLAPACK/scalapack/archive/$COMMIT.zip -o scalapack-$COMMIT.zip
fi
unzip -n -q scalapack-$COMMIT.zip
ln -sf scalapack-$COMMIT scalapack
#ln -sf scalapack-${VERSION} scalapack
#curl -L http://www.netlib.org/scalapack/scalapack-${VERSION}.tgz -o scalapack.tgz
#tar xzf scalapack.tgz
cd scalapack
# macos accelerate does not contain dcombossq
if [[ $(echo "$BLASOPT" |awk '/Accelerate/ {print "Y"; exit}' ) == "Y" ]]; then
    export USE_DCOMBSSQ=1
fi
if [[  -z "$USE_DCOMBSSQ" ]]; then
    patch -p0 -s -N < ../dcombssq.patch
fi
patch -p0 -s -N < ../cmake.patch
#curl -LJO https://github.com/Reference-ScaLAPACK/scalapack/commit/189c84001bcd564296a475c5c757afc9f337e828.patch
#patch -p1 < 189c84001bcd564296a475c5c757afc9f337e828.patch
rm -rf build
mkdir -p build
cd build
if  [[ -n ${FC} ]] &&   [[ ${FC} == xlf ]] || [[ ${FC} == xlf_r ]] || [[ ${FC} == xlf90 ]]|| [[ ${FC} == xlf90_r ]]; then
    Fortran_FLAGS=" -qintsize=4 -qextname "
elif [[ -n ${FC} ]] &&   [[ ${FC} == ftn ]]; then
    if [[ ${PE_ENV} == INTEL ]]; then
	Fortran_FLAGS="-O2 -g -axCORE-AVX2"
    fi
#elif [[ -n ${FC} ]] &&   [[ ${FC} == flang ]]; then
# unset FC=flang since cmake gets lost
#       unset FC
fi
#if [[ ! -z "$BUILD_SCALAPACK"   ]] ; then
#    Fortran_FLAGS+=-I"$NWCHEM_TOP"/src/libext/include
#fi
#fix for clang 12 error in implicit-function-declaration
GOTCLANG=$( "$MPICC" -dM -E - </dev/null 2> /dev/null |grep __clang__|head -1|cut -c19)
if [[ ${GOTCLANG} == "1" ]] ; then
    C_FLAGS=" -Wno-error=implicit-function-declaration "
fi
echo "SCALAPACK_SIZE" is $SCALAPACK_SIZE
if [[ ${FC} == ftn ]]; then
    if [[ ${PE_ENV} == PGI ]]; then
          FC=pgf90
    fi
    if [[ ${PE_ENV} == INTEL ]]; then
	FC=ifort
    fi
    if [[ ${PE_ENV} == GNU ]]; then
	FC=gfortran
    fi
    if [[ ${PE_ENV} == AOCC ]]; then
	FC=flang
    fi
    if [[ ${PE_ENV} == NVIDIA ]]; then
	FC=nvfortran
    fi
    if [[ ${PE_ENV} == CRAY ]]; then
	FC=crayftn
	CC=clang
	#fix for libunwind.so link problem
        export LD_LIBRARY_PATH=/opt/cray/pe/cce/$CRAY_FTN_VERSION/cce-clang/x86_64/lib:/opt/cray/pe/lib64/cce/:$LD_LIBRARY_PATH
    fi
fi
FC_EXTRA=$(${NWCHEM_TOP}/src/config/strip_compiler.sh ${FC})
if [[  -z "$PE_ENV"   ]] ; then
    #check if mpif90 and FC are consistent
    MPIF90_EXTRA=$(${NWCHEM_TOP}/src/config/strip_compiler.sh `${MPIF90} -show`)
    if [[ $MPIF90_EXTRA != $FC_EXTRA ]]; then
        echo which mpif90 is `which mpif90`
        echo mpif90show `${MPIF90} -show`
	echo FC and MPIF90 are not consistent
	echo FC is $FC_EXTRA
	echo MPIF90 is $MPIF90_EXTRA
	exit 1
    fi
fi
if [[  "$SCALAPACK_SIZE" == 8 ]] ; then
    if  [[ ${FC} == f95 ]] || [[ ${FC_EXTRA} == gfortran ]] ; then
    Fortran_FLAGS+=" -fdefault-integer-8 -w "
    elif  [[ ${FC} == xlf ]] || [[ ${FC} == xlf_r ]] || [[ ${FC} == xlf90 ]]|| [[ ${FC} == xlf90_r ]]; then
    Fortran_FLAGS=" -qintsize=8 -qextname "
    elif  [[ ${FC} == crayftn ]]; then
    Fortran_FLAGS=" -s integer64 -h nopattern"
    else
    Fortran_FLAGS+=" -i8 "
    fi
    C_FLAGS+=" -DInt=long"
fi
#skip argument check for gfortran
if  [[ ${FC_EXTRA} == gfortran ]] || [[ ${FC} == f95 ]]; then
    Fortran_FLAGS+=" -fPIC "
    if [[ "$(expr `${FC} -dumpversion | cut -f1 -d.` \> 7)" == 1 ]]; then
	Fortran_FLAGS+=" -std=legacy "
    fi
fi
if [[ ${PE_ENV} == NVIDIA ]] || [[ ${FC} == nvfortran ]] ; then
  Fortran_FLAGS+=" -fPIC "
fi
if [[ "$CRAY_CPU_TARGET" == "mic-knl" ]]; then
    module swap craype-mic-knl craype-haswell
    KNL_SWAP=1
fi
echo compiling with CC="$MPICC"  FC=$MPIF90 CFLAGS="$C_FLAGS" FFLAGS="$Fortran_FLAGS" $CMAKE -Wno-dev ../ -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_C_FLAGS="$C_FLAGS"  -DCMAKE_Fortran_FLAGS="$Fortran_FLAGS" -DTEST_SCALAPACK=OFF  -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=OFF  -DBLAS_openblas_LIBRARY="$BLASOPT"  -DBLAS_LIBRARIES="$BLASOPT"  -DLAPACK_openblas_LIBRARY="$BLASOPT"  -DLAPACK_LIBRARIES="$BLASOPT"
CC="$MPICC"  FC=$MPIF90 CFLAGS="$C_FLAGS" FFLAGS="$Fortran_FLAGS" $CMAKE -Wno-dev ../ -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_C_FLAGS="$C_FLAGS"  -DCMAKE_Fortran_FLAGS="$Fortran_FLAGS" -DTEST_SCALAPACK=OFF  -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=OFF  -DBLAS_openblas_LIBRARY="$BLASOPT"  -DBLAS_LIBRARIES="$BLASOPT"  -DLAPACK_openblas_LIBRARY="$BLASOPT"  -DLAPACK_LIBRARIES="$BLASOPT"
if [[ "$?" != "0" ]]; then
    echo " "
    echo "cmake failed"
    echo " "
    exit 1
fi
make V=0 -j4 scalapack/fast
if [[ "$?" != "0" ]]; then
    echo " "
    echo "compilation failed"
    echo " "
    exit 1
fi
mkdir -p ../../../lib
cp lib/libscalapack.a ../../../lib/libnwc_scalapack.a
if [[ "$KNL_SWAP" == "1" ]]; then
    module swap  craype-haswell craype-mic-knl
fi
