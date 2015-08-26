
add_executable(qperf-read
  main.c
  mrg_GP_socket.c
  mrg_util_error.c
)
target_link_libraries(qperf-read
  dkvstore
)
