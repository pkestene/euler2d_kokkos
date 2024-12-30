# SPDX-FileCopyrightText: 2024 euler2d_kokkos authors
#
# SPDX-License-Identifier: Unlicense

cmake_minimum_required(VERSION 3.22)
project(
  euler2d
  DESCRIPTION
    "euler2d_kokkos is a C++ Kokkos-based CFD miniapp for teaching purpose"
  LANGUAGES CXX C Fortran)

# external dependencies: zlib (required by cnpy)
find_package(ZLIB REQUIRED)

#
# default local cmake macro repository
#
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

# set default build type to "Release"
set(DEFAULT_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(
    STATUS
      "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
  set(CMAKE_BUILD_TYPE
      "${DEFAULT_BUILD_TYPE}"
      CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
                                               "MinSizeRel" "RelWithDebInfo")
endif()

# always export compile commands database
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#
# Kokkos : https://github.com/kokkos/kokkos
#
include(build_or_find_kokkos)

option(USE_DOUBLE "build with double precision" OFF)
if(USE_DOUBLE)
  add_compile_options(-DUSE_DOUBLE)
  # cmake version also have this add_compile_definitions(-DUSE_DOUBLE)
endif()

# HDF5 support (optional)
find_package(HDF5 QUIET)
if(HDF5_FOUND)
  set(HDF5_ENABLED TRUE)
  add_compile_options(-DUSE_HDF5)
else()
  set(HDF5_ENABLED FALSE)
endif()

add_subdirectory(src)

# #################### PRINT CONFIGURE STATUS ######################
message("//===================================================")
message("// ${PROJECT_NAME} build configuration:")
message("//===================================================")
message("")
message("  CMake version          : ${CMAKE_VERSION}")
if(NOT CMAKE_BUILD_TYPE)
  message("  CMake build type       : NOT SET !")
else()
  message("  CMake build type       : ${CMAKE_BUILD_TYPE}")
endif()
message("  CMake install prefix   : ${CMAKE_INSTALL_PREFIX}")
message("  CMake system processor : ${CMAKE_SYSTEM_PROCESSOR}")
message("  CMake system name (OS) : ${CMAKE_SYSTEM_NAME}")
message("")
message("  C++ Compiler : ${CMAKE_CXX_COMPILER_ID} "
        "${CMAKE_CXX_COMPILER_VERSION} " "${CMAKE_CXX_COMPILER_WRAPPER}")
message("                 ${CMAKE_CXX_COMPILER}")

message("  Kokkos version               = ${Kokkos_VERSION}")
message("  Kokkos_CXX_COMPILER          = ${Kokkos_CXX_COMPILER}")
message("  Kokkos_CXX_COMPILER_ID       = ${Kokkos_CXX_COMPILER_ID}")
message("  Kokkos_CXX_STANDARD          = ${Kokkos_CXX_STANDARD}")
message("  Kokkos_OPTIONS               = ${Kokkos_OPTIONS}")
message("  Kokkos_TPLS                  = ${Kokkos_TPLS}")
message("  Kokkos_DIRS                  = ${Kokkos_DIR}")

if(Kokkos_ENABLE_OPENMP)
  message("  Kokkos_ENABLE_OPENMP         = ${Kokkos_ENABLE_OPENMP}")
endif()

if(Kokkos_ENABLE_CUDA)
  message("  Kokkos_ENABLE_CUDA           = ${Kokkos_ENABLE_CUDA}")

  if((${Kokkos_CUDA_LAMBDA_ENABLED}) OR (${Kokkos_ENABLE_CUDA_LAMBDA}))
    message("  Kokkos_ENABLE_CUDA_LAMBDA    = ON")
  else()
    message("  Kokkos_ENABLE_CUDA_LAMBDA    = OFF")
  endif()

  if((${Kokkos_CUDA_CONSTEXPR_ENABLED}) OR (${Kokkos_ENABLE_CUDA_CONSTEXPR}))
    message("  Kokkos_ENABLE_CUDA_CONSTEXPR = ON")
  else()
    message("  Kokkos_ENABLE_CUDA_CONSTEXPR = OFF")
  endif()

  if((${Kokkos_CUDA_UVM_ENABLED}) OR (${Kokkos_ENABLE_CUDA_UVM}))
    message("  Kokkos_ENABLE_CUDA_UVM       = ON")
  else()
    message("  Kokkos_ENABLE_CUDA_UVM       = OFF")
  endif()

  message("  Kokkos CUDA   flags          = ${KOKKOS_CUDA_OPTIONS}")
  # message("  CUDA Compiler      : ${CMAKE_CUDA_COMPILER}")
  #
  # message("  CUDA Compiler exec : ${CUDA_NVCC_EXECUTABLE}")
  #
  # message("  CUDA Compile flags : ${CUDA_NVCC_FLAGS}")
endif(Kokkos_ENABLE_CUDA)

if(Kokkos_ENABLE_HIP)
  message("  Kokkos_ENABLE_HIP            = ${Kokkos_ENABLE_HIP}")
endif(Kokkos_ENABLE_HIP)

if((${Kokkos_TPLS_HWLOC_ENABLED}) OR (${Kokkos_ENABLE_HWLOC}))
  message("  Kokkos_ENABLE_HWLOC          = ON")
else()
  message("  Kokkos_ENABLE_HWLOC          = OFF")
endif()

message("  Kokkos architecture          = ${Kokkos_ARCH}")

message("  HDF5 support                 = ${HDF5_ENABLED}")
