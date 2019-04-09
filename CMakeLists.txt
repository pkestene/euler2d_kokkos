cmake_minimum_required(VERSION 3.1)
project(euler2d CXX C)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(external/kokkos)

option (USE_DOUBLE "build with double precision" ON)
if (USE_DOUBLE)
  add_compile_options(-DUSE_DOUBLE)
  # cmake version also have this
  #add_compile_definitions(-DUSE_DOUBLE)
endif()

add_subdirectory(src)
