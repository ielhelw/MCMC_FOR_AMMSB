configure_file(mcmc/config.h.in mcmc/config.h @ONLY )

find_package(Threads REQUIRED)
find_package(TinyXml2 REQUIRED)

find_package(Boost 1.54.0
  REQUIRED COMPONENTS
  program_options
  iostreams
  system
  filesystem
  thread
)

SET (dkvstore_SRCS )
LIST (APPEND dkvstore_SRCS dkvstore/DKVStoreFile.cc)
if (MCMC_ENABLE_RAMCLOUD)
  LIST (APPEND dkvstore_SRCS dkvstore/DKVStoreRamCloud.cc )
endif(MCMC_ENABLE_RAMCLOUD)
if (MCMC_ENABLE_RDMA)
  LIST (APPEND dkvstore_SRCS dkvstore/DKVStoreRDMA.cc )
  LIST (APPEND dkvstore_SRCS dkvstore/qperf-rdma.c )
endif(MCMC_ENABLE_RDMA)

add_library(dkvstore SHARED
  ${dkvstore_SRCS}
)
target_include_directories(dkvstore PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  # ${GLASSWING_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)
target_link_libraries(dkvstore
  # ${GLASSWING_LIBRARIES}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_THREAD_LIBRARY}
  ${CMAKE_THREAD_LIBS_INIT}
)
if (MCMC_ENABLE_RAMCLOUD)
  target_include_directories(dkvstore PUBLIC
    ${RAMCLOUD_INCLUDE_DIRS}
  )
  target_link_libraries(dkvstore
    ${RAMCLOUD_LIBRARIES}
    ibverbs
    rdmacm
  )
endif(MCMC_ENABLE_RAMCLOUD)
if (MCMC_ENABLE_RDMA)
  target_link_libraries(dkvstore
    ibverbs
    rdmacm
  )
endif(MCMC_ENABLE_RDMA)

SET (mcmc_SRCS )
LIST (APPEND mcmc_SRCS mcmc/np.cc)
LIST (APPEND mcmc_SRCS mcmc/options.cc)
LIST (APPEND mcmc_SRCS mcmc/exception.cc)
LIST (APPEND mcmc_SRCS mcmc/random.cc)
LIST (APPEND mcmc_SRCS mcmc/data.cc)
LIST (APPEND mcmc_SRCS mcmc/network.cc)
LIST (APPEND mcmc_SRCS mcmc/timer.cc)
LIST (APPEND mcmc_SRCS mcmc/preprocess/dataset.cc)
LIST (APPEND mcmc_SRCS mcmc/preprocess/netscience.cc)
LIST (APPEND mcmc_SRCS mcmc/preprocess/relativity.cc)
LIST (APPEND mcmc_SRCS mcmc/preprocess/data_factory.cc)
LIST (APPEND mcmc_SRCS mcmc/learning/learner.cc)
LIST (APPEND mcmc_SRCS mcmc/learning/mcmc_sampler_stochastic.cc)
if (MCMC_ENABLE_DISTRIBUTED)
  LIST (APPEND mcmc_SRCS mcmc/learning/mcmc_sampler_stochastic_distr.cc)
endif(MCMC_ENABLE_DISTRIBUTED)

add_library(mcmc SHARED
  ${mcmc_SRCS}
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
  ${CMAKE_THREAD_LIBS_INIT}
  dkvstore
)
