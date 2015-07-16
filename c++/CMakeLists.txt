cmake_minimum_required(VERSION 2.8)

SET(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

LIST(APPEND CMAKE_CXX_FLAGS " -std=c++11 ")

add_subdirectory(src)
add_subdirectory(apps)
