cmake_minimum_required(VERSION 3.10)
project(
  euler2d
  DESCRIPTION "euler2d_kokkos is mini application using C++/Kokkos library, used for teaching purpose"
  LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

# build kokkos, provided as a git submodule
add_subdirectory(external/kokkos)

option (USE_DOUBLE "build with double precision" ON)
if (USE_DOUBLE)
  add_compile_options(-DUSE_DOUBLE)
  # cmake version also have this
  #add_compile_definitions(-DUSE_DOUBLE)
endif()

add_subdirectory(src)
