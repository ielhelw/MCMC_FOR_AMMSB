cmake_minimum_required(VERSION 2.8)

find_package(Boost 1.54.0
  REQUIRED COMPONENTS
  program_options
)


LIST(APPEND CMAKE_CXX_FLAGS " -std=c++11 ")

add_library(mcmc STATIC
  mcmc/np.cc
  mcmc/options.cc
  mcmc/exception.cc
  mcmc/random.cc
  mcmc/estimate_phi.cc
  mcmc/data.cc
  mcmc/network.cc
  mcmc/sample_latent_vars.cc
  mcmc/preprocess/dataset.cc
  mcmc/preprocess/netscience.cc
  mcmc/preprocess/relativity.cc
  mcmc/preprocess/data_factory.cc
)
target_include_directories(mcmc PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${Boost_INCLUDE_DIRS}
)
target_link_libraries(mcmc
  ${Boost_LIBRARIES}
)

