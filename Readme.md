[![Build Status](https://travis-ci.org/pkestene/euler2d_kokkos.svg?branch=master)](https://travis-ci.org/pkestene/euler2d_kokkos)

![C/C++ CI](https://github.com/pkestene/euler2d_kokkos/workflows/C/C++%20CI/badge.svg)

![blast2d_1024x1536](https://github.com/pkestene/euler2d_kokkos/blob/master/blast2d.gif)

This miniapp solves the [compressible fluid dynamics (Euler) equations](https://en.wikipedia.org/wiki/Euler_equations_(fluid_dynamics)) using 2D cartesian grids, parallelized for shared memory system. The full application for 2D/3D grids with [MPI](https://www.mpi-forum.org/) and [Kokkos](https://github.com/kokkos/kokkos) is available here : https://github.com/pkestene/euler_kokkos

# Get the source

Our miniApp uses kokkos as a git submodule, configured to use the `develop` branch of kokkos.
In order to download the sources all-in-one (miniApp + kokkos):

```shell
git clone git@github.com:pkestene/euler2d_kokkos.git
```

# Build

We strongly recommend the out-of-source build, so that one can have one build directory per architecture.

## If you already have Kokkos installed

Just make sure that your env variable `CMAKE_PREFIX_PATH` point to the location where Kokkos where installed. More precisely if Kokkos is installed in `KOKKOS_ROOT`, you add `$KOKKOS_ROOT/lib/cmake` to your `CMAKE_PREFIX_PATH`; this way kokkos will be found automagically by cmake, and the right Kokkos backend will be selected.

```shell
mkdir -p build; cd build
cmake -DEULER2D_KOKKOS_BUILD=OFF ..
make
```

You should now have executable *euler2d*. You can run a simply implode test like this
```shell
cd src
./euler2d ./test_implode.ini
```


## Kokkos is not already installed => build for Kokkos/OpenMP

To build for Kokkos/OpenMP backend (which is the default backend):
```shell
mkdir -p build/openmp; cd build/openmp
cmake -DEULER2D_KOKKOS_BUILD=ON -DEULER2D_KOKKOS_BACKEND=OpenMP ../..
make
```

You should now have executable *euler2d*. You can run a simply implode test like this
```shell
cd src
./euler2d ./test_implode.ini
```

Optionally, you can (recommended) activate HWLOC support by turning ON the flag Kokkos_ENABLE_HWLOC.


## Kokkos is not already installed => build for Kokkos/CUDA

Obviously, you need to have Nvidia/CUDA driver and toolkit installed on your platform.

Example configuration:
```shell
mkdir build/cuda; cd build/cuda
cmake -DEULER2D_KOKKOS_BUILD=ON -DEULER2D_KOKKOS_BACKEND=Cuda -DKokkos_ENABLE_CUDA_LAMBDA=ON -DKokkos_ENABLE_HWLOC=ON ../..
make
```

Be aware that, kokkos will not use Nvidia compiler `nvcc` directly, but will use a wrapper instead (located in Kokkos source directory).

then `make` should give you a working executable `euler2d` running on GPU.

```shell
cd src
./euler2d ./test_implode.ini
```
