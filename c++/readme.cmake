module load boost-1.54
THIRDPARTY=/home/rutger/projects/greenclouds/3rdparty

mkdir build
cd build
cmake \
	-DCMAKE_C_COMPILER=`which gcc` \
	-DCMAKE_CXX_COMPILER=`which g++` \
	-DBOOST_ROOT=`dirname $BOOST_INCLUDE` \
	-DGTEST_ROOT=$THIRDPARTY/gtest-1.7.0 \
	-DTINYXML2_ROOT=$THIRDPARTY/tinyxml2 \
	-DRAMCLOUD_ROOT=$THIRDPARTY/ramcloud \
	-DSPARSEHASH_ROOT=$THIRDPARTY/google-sparsehash \
	-DMCMC_ENABLE_RDMA=ON \
	..
