# find_package(Glasswing REQUIRED)
find_package(RamCloud REQUIRED)
find_package(TinyXml2 REQUIRED)
find_package(SparseHash REQUIRED)

find_package(Boost 1.54.0
  REQUIRED COMPONENTS
  program_options
  iostreams
  system
  filesystem
  thread
)

add_library(dkvstore SHARED
  dkvstore/DKVStoreRamCloud.cc
  dkvstore/DKVStoreFile.cc
  dkvstore/DKVStoreRDMA.cc
  dkvstore/qperf-rdma.c
)
target_include_directories(dkvstore PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  # ${GLASSWING_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${RAMCLOUD_INCLUDE_DIRS}
)
target_link_libraries(dkvstore
  ${RAMCLOUD_LIBRARIES}
  # ${GLASSWING_LIBRARIES}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_THREAD_LIBRARY}
)

option( MCMC_XORSHIFT_ENABLE "Enable Random:xorshift" ON )
if (MCMC_XORSHIFT_ENABLE)
else(MCMC_XORSHIFT_ENABLE)
  SET(MCMC_RANDOM_SYSTEM "1")
endif(MCMC_XORSHIFT_ENABLE)

option ( MCMC_GOOGLE_SPARSE_HASH_ENABLE "Enable GoogleSparseHash" ON )
if (MCMC_GOOGLE_SPARSE_HASH_ENABLE)
  SET(MCMC_USE_GOOGLE_SPARSE_HASH "1")
endif(MCMC_GOOGLE_SPARSE_HASH_ENABLE)

option (MCMC_EDGESET_IS_ADJACENCY_LIST "Use adjacency list graph representation" ON)
if (MCMC_EDGESET_IS_ADJACENCY_LIST)
  SET(MCMC_EDGESET_IS_ADJACENCY_LIST "1")
endif(MCMC_EDGESET_IS_ADJACENCY_LIST)

configure_file(mcmc/config.h.in mcmc/config.h @ONLY )

add_library(mcmc SHARED
  mcmc/np.cc
  mcmc/options.cc
  mcmc/exception.cc
  mcmc/random.cc
  mcmc/data.cc
  mcmc/network.cc
  mcmc/timer.cc
  mcmc/preprocess/dataset.cc
  mcmc/preprocess/netscience.cc
  mcmc/preprocess/relativity.cc
  mcmc/preprocess/data_factory.cc
  mcmc/learning/learner.cc
  mcmc/learning/mcmc_sampler_stochastic.cc
  mcmc/learning/mcmc_sampler_stochastic_distr.cc
)

target_include_directories(mcmc PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${Boost_INCLUDE_DIRS}
  ${TINYXML2_INCLUDE_DIRS}
  ${SPARSEHASH_INCLUDE_DIRS}
)
target_link_libraries(mcmc
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_IOSTREAMS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_THREAD_LIBRARY}
  ${TINYXML2_LIBRARIES}
  dkvstore
)
