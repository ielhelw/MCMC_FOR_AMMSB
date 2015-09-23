
THIRDPARTY=`(cd ../../../../3rdparty; pwd)`

# On das4:
module load boost-1.54
SPARSEHASH_ROOT=$THIRDPARTY/google-sparsehash

# On das5:
export BOOST_INCLUDE="$(dirname $(dirname $(which gcc)))/include"
SPARSEHASH_ROOT=/usr

mkdir build-seq
cd build-seq
cmake \
        -DCMAKE_C_COMPILER=`which gcc` \
        -DCMAKE_CXX_COMPILER=`which g++` \
        -DBOOST_ROOT=`dirname $BOOST_INCLUDE` \
        -DGTEST_ROOT=$THIRDPARTY/gtest-1.7.0 \
        -DTINYXML2_ROOT=$THIRDPARTY/tinyxml2 \
        -DSPARSEHASH_ROOT=$SPARSEHASH_ROOT \
        -DMCMC_BUILD_MODE=SEQ \
        ..
cd ..

mkdir build-distr
cd build-distr
cmake \
        -DCMAKE_C_COMPILER=`which mpicc` \
        -DCMAKE_CXX_COMPILER=`which mpicxx` \
        -DBOOST_ROOT=`dirname $BOOST_INCLUDE` \
        -DGTEST_ROOT=$THIRDPARTY/gtest-1.7.0 \
        -DTINYXML2_ROOT=$THIRDPARTY/tinyxml2 \
        -DSPARSEHASH_ROOT=$SPARSEHASH_ROOT \
        -DMCMC_BUILD_MODE=DISTR \
# optionally:
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_BUILD_TYPE=Debug \
        ..
cd ..

mkdir build-compat
cd build-compat
cmake \
        -DCMAKE_C_COMPILER=`which gcc` \
        -DCMAKE_CXX_COMPILER=`which g++` \
        -DBOOST_ROOT=`dirname $BOOST_INCLUDE` \
        -DGTEST_ROOT=$THIRDPARTY/gtest-1.7.0 \
        -DTINYXML2_ROOT=$THIRDPARTY/tinyxml2 \
        -DMCMC_BUILD_MODE=COMPAT \
        ..
cd ..
