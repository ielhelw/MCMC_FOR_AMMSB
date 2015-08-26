
add_executable(rand-distr
  main.cc
)
target_link_libraries(rand-distr
  mcmc
  dkvstore
)
