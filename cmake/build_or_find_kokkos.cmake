# Two variants:
# 1. If user set EULER2D_KOKKOS_BUILD, we use kokkos sources from submodule and build kokkos
# 2. If EULER2D_KOKKOS_BUILD=OFF, don't build kokkos, but use find_package for setup (must have kokkos already installed)

#
# Does euler2d builds kokkos (https://github.com/kokkos/kokkos) ?
#
option(EULER2D_KOKKOS_BUILD "Turn ON if you want to build kokkos (default: OFF)" OFF)


# raise the minimum C++ standard level if not already done
# when build kokkos, it defaults to c++-14
# when using installed kokkos, it is not set, so defaulting to c++-14
# should be ok until kokkos requires c++-17 or later
if (NOT "${CMAKE_CXX_STANDARD}")
  set(CMAKE_CXX_STANDARD 17)
endif()

# check if user requested a build of kokkos
if(EULER2D_KOKKOS_BUILD)

  message("[euler2d / kokkos] Building kokkos from source")

  # build kokkos, provided as a git submodule
  add_subdirectory(external/kokkos)

  set(EULER2D_KOKKOS_BUILTIN TRUE)

else()

  #
  # check if already installed kokkos exists
  #
  find_package(Kokkos 3.7.00 REQUIRED)

  if(TARGET Kokkos::kokkos)

    kokkos_check( DEVICES "OpenMP" )

    if(EULER2D_ENABLE_GPU_CUDA)
      # kokkos_check is defined in KokkosConfigCommon.cmake
      kokkos_check( DEVICES "Cuda" )
      kokkos_check( OPTIONS CUDA_LAMBDA)
      kokkos_check( OPTIONS CUDA_CONSTEXPR)
    elseif(EULER2D_ENABLE_GPU_HIP)
      # TODO
      kokkos_check( DEVICES "HIP" )
    endif()

    message("[euler2d / kokkos] Kokkos found via find_package")
    set(EULER2D_KOKKOS_FOUND True)
    set(HAVE_KOKKOS 1)

  else()

    message(FATAL_ERROR "[euler2d / kokkos] Kokkos is required but not found by find_package. Please adjet your env variable CMAKE_PREFIX_PATH (or Kokkos_ROOT) to where Kokkos is installed on your machine !")

  endif()

endif()
