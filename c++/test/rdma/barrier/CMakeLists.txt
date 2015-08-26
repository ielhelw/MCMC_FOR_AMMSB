
add_executable(rdma-barrier
  main.cc
)
target_link_libraries(rdma-barrier
  dkvstore
  mcmc
)
