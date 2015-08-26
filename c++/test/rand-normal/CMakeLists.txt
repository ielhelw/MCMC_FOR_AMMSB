
add_executable(rand-normal
  main.cc
)
target_link_libraries(rand-normal
  mcmc
  dkvstore
)
