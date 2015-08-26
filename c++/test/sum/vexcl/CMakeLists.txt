
add_executable(vexcl-sum
  vexcl-compute-test.cc
)
target_link_libraries(vexcl-sum
  mcmc
  dkvstore
)
