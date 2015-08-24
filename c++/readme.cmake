mkdir build

cd build
cmake ..

cmake -DCMAKE_C_COMPILER=`which gcc` ..
cmake -DCMAKE_CXX_COMPILER=`which g++` ..

module load boost-1.54
cmake -DBOOST_ROOT=`dirname $BOOST_INCLUDE`  ..

THIRDPARTY=/home/rutger/projects/greenclouds/3rdparty
cmake -DGTEST_ROOT=$THIRDPARTY/gtest-1.7.0 ..
cmake -DTINYXML2_ROOT=$THIRDPARTY/tinyxml2 ..
cmake -DRAMCLOUD_ROOT=$THIRDPARTY/ramcloud ..
cmake -DSPARSEHASH_ROOT=$THIRDPARTY/google-sparsehash ..
