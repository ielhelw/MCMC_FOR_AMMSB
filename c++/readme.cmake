module load boost-1.54
THIRDPARTY=`(cd ../../../../3rdparty; pwd)`

mkdir build
cd build
cmake \
	-DCMAKE_C_COMPILER=`which gcc` \
	-DCMAKE_CXX_COMPILER=`which g++` \
	-DBOOST_ROOT=`dirname $BOOST_INCLUDE` \
	-DGTEST_ROOT=$THIRDPARTY/gtest-1.7.0 \
	-DTINYXML2_ROOT=$THIRDPARTY/tinyxml2 \
	-DSPARSEHASH_ROOT=$THIRDPARTY/google-sparsehash \
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
	-DRAMCLOUD_ROOT=$THIRDPARTY/ramcloud \
	-DSPARSEHASH_ROOT=$THIRDPARTY/google-sparsehash \
	-DMCMC_BUILD_MODE=DISTR \
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
