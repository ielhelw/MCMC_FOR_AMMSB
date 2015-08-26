
add_executable(vexcl-scans
  vexcl-compute-test.cc
)
target_link_libraries(vexcl-scans
  mcmc
  dkvstore
)
