
add_executable(dkv-read-only
  main.cc
)
target_link_libraries(dkv-read-only
  mcmc
  dkvstore
)
