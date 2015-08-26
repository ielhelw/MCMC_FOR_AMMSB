
add_executable(testOOBNetwork
  testOOBNetwork.cc
)
target_link_libraries(testOOBNetwork
  dkvstore
  mcmc
)
