find_package(Glasswing REQUIRED)
find_package(RamCloud REQUIRED)
find_package(TinyXml2 REQUIRED)

find_package(Boost 1.54.0
  REQUIRED COMPONENTS
  program_options
  iostreams
  system
)

add_library(mcmc SHARED
  mcmc/np.cc
  mcmc/options.cc
  mcmc/exception.cc
  mcmc/random.cc
  mcmc/estimate_phi.cc
  mcmc/data.cc
  mcmc/network.cc
  mcmc/timer.cc
  mcmc/sample_latent_vars.cc
  mcmc/preprocess/dataset.cc
  mcmc/preprocess/netscience.cc
  mcmc/preprocess/relativity.cc
  mcmc/preprocess/data_factory.cc
  mcmc/learning/learner.cc
  mcmc/learning/mcmc_sampler_stochastic.cc
)
target_include_directories(mcmc PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${Boost_INCLUDE_DIRS}
)
target_link_libraries(mcmc
  ${Boost_LIBRARIES}
  ${TINYXML2_LIBRARIES}
)

add_library(dkvstore SHARED
  dkvstore/DKVStoreRamCloud.cc
  dkvstore/DKVStoreFile.cc
  #  dkvstore/DKVStoreRDMA.cc
)
target_include_directories(dkvstore PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${GLASSWING_INCLUDE_DIRS}
  ${RAMCLOUD_INCLUDE_DIRS}
)
target_link_libraries(dkvstore
  mcmc
  ${RAMCLOUD_LIBRARIES}
  ${GLASSWING_LIBRARIES}
)
