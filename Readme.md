[![Build Status](https://travis-ci.org/pkestene/euler2d_kokkos.svg?branch=master)](https://travis-ci.org/pkestene/euler2d_kokkos)

![C/C++ CI](https://github.com/pkestene/euler2d_kokkos/workflows/C/C++%20CI/badge.svg)

![blast2d_1024x1536](https://github.com/pkestene/euler2d_kokkos/blob/master/blast2d.gif)

This miniapp solves the [compressible fluid dynamics (Euler) equations](https://en.wikipedia.org/wiki/Euler_equations_(fluid_dynamics)) using 2D cartesian grids, parallelized for shared memory system. The full application for 2D/3D grids with [MPI](https://www.mpi-forum.org/) and [Kokkos](https://github.com/kokkos/kokkos) is available here : https://github.com/pkestene/euler_kokkos

# Get the source

Our miniApp uses kokkos as a git submodule, configured to use the `develop` branch of kokkos.
In order to download the sources all-in-one (miniApp + kokkos):

```shell
git clone --recursive git@github.com:pkestene/euler2d_kokkos.git
```

Alternatively, if you didn't use option `--recursive`, you can afterwards retrieve kokkos like this
```shell
git submodule init
git submodule update
```

# Build

We strongly recommend the out-of-source build, so that one can have one build directory per architecture.

## build for Kokkos/OpenMP

To build for Kokkos/OpenMP backend (which is the default backend):
```shell
mkdir build_openmp; cd build_openmp
cmake -DKokkos_ENABLE_OPENMP=ON ..
make
```

You should now have executable *euler2d*. You can run a simply implode test like this
```shell
cd src
./euler2d ./test_implode.ini
```

Optionally, you can (recommended) activate HWLOC support by turning ON the flag Kokkos_ENABLE_HWLOC.


## build for Kokkos/CUDA

Obviously, you need to have Nvidia/CUDA driver and toolkit installed on your platform.
Then you need to
 1. tell cmake to use kokkos compiler wrapper for cuda:
    ```shell
    export CXX=/complete/path/to/kokos/bin/nvcc_wrapper
    ```
 2. activate CUDA backend in the ccmake interface. 
    * Just turn on Kokkos_ENABLE_CUDA 
    * select cuda architecture, e.g. set Kokkos_ARCH_KEPLER37=ON (for Nvidia K80 boards)

then `make` should give you a working executable `euler2d` running on GPU.

```shell
cd src
./euler2d ./test_implode.ini
```

