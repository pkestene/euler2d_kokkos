cmake_minimum_required(VERSION 3.1)
project(euler2d CXX C)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
list(APPEND CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -O3)

add_subdirectory(kokkos)

if(KOKKOS_ENABLE_CUDA)
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUDA" )
endif()

option (USE_DOUBLE "build with double precision" ON)
if (USE_DOUBLE)
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_DOUBLE" )
endif()

add_executable(euler2d
  config/inih/ini.cpp
  config/inih/INIReader.cpp
  config/ConfigMap.cpp
  HydroParams.cpp
  HydroRun.cpp
  Timer.cpp
  main.cpp)
target_include_directories(euler2d
  PUBLIC
  ${Kokkos_INCLUDE_DIRS_RET})
target_link_libraries(euler2d kokkos)
