
add_executable(random
  test_random.cc
)
target_link_libraries(random
  mcmc
  dkvstore
)
