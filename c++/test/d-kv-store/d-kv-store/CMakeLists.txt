
add_executable(d-kv-store
  main.cc
)
target_link_libraries(d-kv-store
  mcmc
  dkvstore
)
