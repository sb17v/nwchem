#!/usr/bin/env bash
echo "start compile"
set -e
# source env. variables
if [[ -z "$TRAVIS_BUILD_DIR" ]] ; then
    TRAVIS_BUILD_DIR=$(pwd)
fi
echo TRAVIS_BUILD_DIR is $TRAVIS_BUILD_DIR
source $TRAVIS_BUILD_DIR/travis/nwchem.bashrc
echo ============================================================
env|egrep BLAS     || true
env|egrep USE_6   || true
ls -lrt $TRAVIS_BUILD_DIR|tail -3
echo ============================================================
os=`uname`
arch=`uname -m`
if [[ "$NWCHEM_MODULES" == "tce" ]]; then 
    export EACCSD=1
    export IPCCSD=1
fi
cd $TRAVIS_BUILD_DIR/src
#FDOPT="-O0 -g"
if [[ "$arch" == "aarch64" ]]; then 
    if [[ "$FC" == "flang" ]]  ; then
	export BUILD_MPICH=1
        FOPT="-O2  -ffast-math"
    elif [[ "$(basename -- $FC | cut -d \- -f 1)" == "nvfortran" ]] ; then
	export USE_FPICF=1
#	export MPICH_FC=nvfortran
	export MPICH_FC=$FC
	env|egrep FC
    else
#should be gfortran	
	if [[ "$NWCHEM_MODULES" == "tce" ]]; then 
	    FOPT="-O0 -fno-aggressive-loop-optimizations"
	else
	    FOPT="-O1 -fno-aggressive-loop-optimizations"
	fi
    fi
else
    if [[ "$FC" == "ifort" ]] || [[ "$FC" == "ifx" ]] ; then
	FOPT=-O2
	if [[ "$os" == "Darwin" ]]; then
	    export BUILD_MPICH=1
 	    export BLASOPT="-L$MKLROOT  -Wl,-rpath,${MKLROOT}/lib -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core  -lpthread -lm -ldl"
	else
	    export USE_FPICF=Y
 	    export BLASOPT="-L$MKLROOT -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core  -lpthread -lm -ldl"
	    export SCALAPACK_LIB="-L$MKLROOT -lmkl_scalapack_ilp64 -lmkl_blacs_intelmpi_ilp64 -lpthread -lm -ldl"
	    export SCALAPACK_SIZE=8
	    unset BUILD_SCALAPACK
	fi
        unset BUILD_OPENBLAS
	export BLAS_SIZE=8
	export LAPACK_LIB="$BLASOPT"
	export I_MPI_F90="$FC"
    elif [[ "$FC" == "flang" ]] || [[ "$(basename -- $FC | cut -d \- -f 1)" == "nvfortran" ]] ; then
	export BUILD_MPICH=1
        if [[ "$FC" == "flang" ]]; then
	    FOPT="-O2  -ffast-math"
	fi
        if [[ "$FC" == "nvfortran" ]]; then
	    export USE_FPICF=1
#	    FOPT="-O2 -tp haswell"
	fi
    fi
fi    
 if [[ "$os" == "Darwin" ]]; then 
   if [[ "$NWCHEM_MODULES" == "tce" ]]; then
     FOPT="-O1 -fno-aggressive-loop-optimizations"
   fi
   if [[ ! -z "$USE_SIMINT" ]] ; then 
       FOPT="-O0 -fno-aggressive-loop-optimizations"
       SIMINT_BUILD_TYPE=Debug
       export PATH="/usr/local/bin:$PATH"
#       export LDFLAGS="-L/usr/local/opt/python@3.7/lib:$LDFLAGS"
   fi
   if [[ -z "$TRAVIS_HOME" ]]; then
       env
       if [[ -z "$FOPT" ]]; then
	   make V=0   -j3
       else
	   make V=0 FOPTIMIZE="$FOPT"   -j3
       fi
   else
       ../travis/sleep_loop.sh make V=1 FOPTIMIZE="$FOPT"   -j3
   fi
     cd $TRAVIS_BUILD_DIR/src/64to32blas 
     make
     cd $TRAVIS_BUILD_DIR/src
     ../contrib/getmem.nwchem 1000
     otool -L ../bin/MACX64/nwchem
#     printenv DYLD_LIBRARY_PATH
#     ls -lrt $DYLD_LIBRARY_PATH
#      tail -120 make.log
 elif [[ "$os" == "Linux" ]]; then
     export MAKEFLAGS=-j3
     echo    "$FOPT$FDOPT"
if [[ -z "$TRAVIS_HOME" ]]; then
    if [[ -z "$FOPT" ]]; then
	make V=0   -j3
    else
	make V=0 FOPTIMIZE="$FOPT"   -j3
    fi
else
    ../travis/sleep_loop.sh make V=1 FOPTIMIZE="$FOPT"  -j3
fi
     cd $TRAVIS_BUILD_DIR/src/64to32blas 
     make
     cd $TRAVIS_BUILD_DIR/src
     $TRAVIS_BUILD_DIR/contrib/getmem.nwchem 1000
 fi
 #caching
 mkdir -p $TRAVIS_BUILD_DIR/.cachedir/binaries/$NWCHEM_TARGET $TRAVIS_BUILD_DIR/.cachedir/files
 cp $TRAVIS_BUILD_DIR/bin/$NWCHEM_TARGET/nwchem  $NWCHEM_EXECUTABLE
 echo === ls binaries cache ===
 ls -lrt $TRAVIS_BUILD_DIR/.cachedir/binaries/$NWCHEM_TARGET/ 
 echo =========================
 rsync -av $TRAVIS_BUILD_DIR/src/basis/libraries  $TRAVIS_BUILD_DIR/.cachedir/files/.
 rsync -av $TRAVIS_BUILD_DIR/src/nwpw/libraryps  $TRAVIS_BUILD_DIR/.cachedir/files/.
