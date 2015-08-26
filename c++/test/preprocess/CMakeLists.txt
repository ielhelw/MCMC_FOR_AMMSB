
add_executable(preprocess
  main.cc
)
target_link_libraries(preprocess
  mcmc
  dkvstore
)
