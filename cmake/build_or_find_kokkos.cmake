# Two variants:
# 1. If user set EULER2D_KOKKOS_BUILD, we download kokkos sources and build them using FetchContent (which actually uses add_subdirectory)
# 2. If EULER2D_KOKKOS_BUILD=OFF, don't build kokkos, but use find_package for setup (you must have kokkos already installed)

#
# Do we want to build kokkos (https://github.com/kokkos/kokkos) ?
#
option(EULER2D_KOKKOS_BUILD "Turn ON if you want to build kokkos (default: OFF)" OFF)

#
# Option to use git (instead of tarball release) for downloading kokkos
#
option(EULER2D_KOKKOS_USE_GIT "Turn ON if you want to use git to download Kokkos sources (default: OFF)" OFF)

#
# Options to specify target device backend
#

# set default backend
set(EULER2D_KOKKOS_BACKEND "Undefined" CACHE STRING
  "Kokkos default backend device")

# Set the possible values for kokkos backend device
set_property(CACHE EULER2D_KOKKOS_BACKEND PROPERTY STRINGS
  "OpenMP" "Cuda" "HIP" "Undefined")


# raise the minimum C++ standard level if not already done
# when build kokkos, it defaults to c++-17
# when using installed kokkos, it is not set, so defaulting to c++-17
# kokkos 4.0.00 requires c++-17 anyway
if (NOT "${CMAKE_CXX_STANDARD}")
  set(CMAKE_CXX_STANDARD 17)
endif()

# check if user requested a build of kokkos
if(EULER2D_KOKKOS_BUILD)

  message("[euler2d / kokkos] Building kokkos from source")


  # Kokkos default build options

  # set install path
  list (APPEND EULER2D_KOKKOS_CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${KOKKOS_INSTALL_DIR})

  # use predefined cmake args
  # can be override on the command line
  if (EULER2D_KOKKOS_BACKEND MATCHES "Cuda")

    if ((NOT DEFINED Kokkos_ENABLE_HWLOC) OR (NOT Kokkos_ENABLE_HWLOC))
      set(Kokkos_ENABLE_HWLOC ON)
    endif()

    if ((NOT DEFINED Kokkos_ENABLE_OPENMP) OR (NOT Kokkos_ENABLE_OPENMP))
      set(Kokkos_ENABLE_OPENMP ON)
    endif()

    if ((NOT DEFINED Kokkos_ENABLE_CUDA) OR (NOT Kokkos_ENABLE_CUDA))
      set(Kokkos_ENABLE_CUDA ON)
    endif()

    if ((NOT DEFINED Kokkos_ENABLE_CUDA_LAMBDA) OR (NOT Kokkos_ENABLE_CUDA_LAMBDA))
      set(Kokkos_ENABLE_CUDA_LAMBDA ON)
    endif()

    if ((NOT DEFINED Kokkos_ENABLE_CUDA_CONSTEXPR) OR (NOT Kokkos_ENABLE_CUDA_CONSTEXPR))
      set(Kokkos_ENABLE_CUDA_CONSTEXPR ON)
    endif()

    # Note : cuda architecture will probed by kokkos cmake configure

  elseif(EULER2D_KOKKOS_BACKEND MATCHES "OpenMP")

    if ((NOT DEFINED Kokkos_ENABLE_HWLOC) OR (NOT Kokkos_ENABLE_HWLOC))
      set(Kokkos_ENABLE_HWLOC ON)
    endif()

    if ((NOT DEFINED Kokkos_ENABLE_OPENMP) OR (NOT Kokkos_ENABLE_OPENMP))
      set(Kokkos_ENABLE_OPENMP ON)
    endif()

  elseif(EULER2D_KOKKOS_BACKEND MATCHES "HIP")

    if ((NOT DEFINED Kokkos_ENABLE_HWLOC) OR (NOT Kokkos_ENABLE_HWLOC))
      set(Kokkos_ENABLE_HWLOC ON)
    endif()

    if ((NOT DEFINED Kokkos_ENABLE_OPENMP) OR (NOT Kokkos_ENABLE_OPENMP))
      set(Kokkos_ENABLE_OPENMP ON)
    endif()

    if ((NOT DEFINED Kokkos_ENABLE_HIP) OR (NOT Kokkos_ENABLE_HIP))
      set(Kokkos_ENABLE_HIP ON)
    endif()

  elseif(EULER2D_KOKKOS_BACKEND MATCHES "Undefined")

    message(FATAL_ERROR "[euler2d / kokkos] You must chose a valid EULER2D_KOKKOS_BACKEND !")

  endif()

  #find_package(Git REQUIRED)
  include (FetchContent)

  if (EULER2D_KOKKOS_USE_GIT)
    FetchContent_Declare( kokkos_external
      GIT_REPOSITORY https://github.com/kokkos/kokkos.git
      GIT_TAG 4.0.00
      )
  else()
    FetchContent_Declare( kokkos_external
      URL https://github.com/kokkos/kokkos/archive/refs/tags/4.0.00.tar.gz
      )
  endif()

  # Import kokkos targets (download, and call add_subdirectory)
  FetchContent_MakeAvailable(kokkos_external)

  if(TARGET Kokkos::kokkos)
    message("[euler2d / kokkos] Kokkos found (using FetchContent)")
    set(EULER2D_KOKKOS_FOUND True)
    set(HAVE_KOKKOS 1)
  else()
    message("[euler2d / kokkos] we shouldn't be here. We've just integrated kokkos build into euler2d build !")
  endif()

  set(EULER2D_KOKKOS_BUILTIN TRUE)

else()

  #
  # check if an already installed kokkos exists
  #
  find_package(Kokkos 3.7.00 REQUIRED)

  if(TARGET Kokkos::kokkos)

    # kokkos_check is defined in KokkosConfigCommon.cmake
    kokkos_check( DEVICES "OpenMP" RETURN_VALUE KOKKOS_DEVICE_ENABLE_OPENMP)
    kokkos_check( DEVICES "Cuda" RETURN_VALUE KOKKOS_DEVICE_ENABLE_CUDA)
    kokkos_check( DEVICES "HIP" RETURN_VALUE KOKKOS_DEVICE_ENABLE_HIP)

    kokkos_check( TPLS "HWLOC" RETURN_VALUE Kokkos_TPLS_HWLOC_ENABLED)

    if(KOKKOS_DEVICE_ENABLE_CUDA)
      set(EULER2D_KOKKOS_BACKEND "Cuda")
      kokkos_check( OPTIONS CUDA_LAMBDA RETURN_VALUE Kokkos_CUDA_LAMBDA_ENABLED)
      kokkos_check( OPTIONS CUDA_CONSTEXPR RETURN_VALUE Kokkos_CUDA_CONSTEXPR_ENABLED)
      kokkos_check( OPTIONS CUDA_UVM RETURN_VALUE Kokkos_CUDA_UVM_ENABLED)
    elseif(KOKKOS_DEVICE_ENABLE_HIP)
      set(EULER2D_KOKKOS_BACKEND "HIP")
    elseif(KOKKOS_DEVICE_ENABLE_OPENMP)
      set(EULER2D_KOKKOS_BACKEND "OpenMP")
    endif()

    message("[euler2d / kokkos] Kokkos found via find_package; default backend is ${EULER2D_KOKKOS_BACKEND}")
    set(EULER2D_KOKKOS_FOUND True)
    set(HAVE_KOKKOS 1)

  else()

    message(FATAL_ERROR "[euler2d / kokkos] Kokkos is required but not found by find_package. Please adjust your env variable CMAKE_PREFIX_PATH (or Kokkos_ROOT) to where Kokkos is installed on your machine !")

  endif()

endif()