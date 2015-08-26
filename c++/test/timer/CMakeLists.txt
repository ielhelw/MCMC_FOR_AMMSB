
add_executable(timer
  main.cc
)
target_link_libraries(timer
  mcmc
  dkvstore
)
