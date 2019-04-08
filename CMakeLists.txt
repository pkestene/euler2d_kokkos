cmake_minimum_required(VERSION 3.1)
project(euler2d CXX C)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(external/kokkos)

option (USE_DOUBLE "build with double precision" ON)
if (USE_DOUBLE)
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_DOUBLE" )
endif()

add_subdirectory(src)
