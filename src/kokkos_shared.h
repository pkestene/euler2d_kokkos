#ifndef KOKKOS_SHARED_H_
#define KOKKOS_SHARED_H_

#include <Kokkos_Core.hpp>

#include "real_type.h"
#include "hydro_common.h"

namespace euler2d
{

// using Device = Kokkos::DefaultExecutionSpace;

// first index is space location, second is hydro variable
// number of hydro variables is 4 in 2D, 5 in 3D
template <typename device_t>
using DataArray = Kokkos::View<real_t ** [NBVAR], device_t>;

template <typename device_t>
using DataArrayHost = typename DataArray<device_t>::HostMirror;

/// a POD data structure to store local conservative / primitive variables
using HydroState = Kokkos::Array<real_t, NBVAR>;

/**
 * Retrieve cartesian coordinate from index, using memory layout information.
 *
 * for each execution space define a preferred layout.
 * Prefer left layout  for CUDA execution space.
 * Prefer right layout for OpenMP execution space.
 */
KOKKOS_INLINE_FUNCTION
void
index2coord(int index, int & i, int & j, int Nx, int Ny)
{
#ifdef KOKKOS_ENABLE_CUDA
  j = index / Nx;
  i = index - j * Nx;
#else
  i = index / Ny;
  j = index - i * Ny;
#endif
}

KOKKOS_INLINE_FUNCTION
int
coord2index(int i, int j, int Nx, int Ny)
{
#ifdef KOKKOS_ENABLE_CUDA
  return i + Nx * j; // left layout
#else
  return j + Ny * i; // right layout
#endif
}

} // namespace euler2d

#endif // KOKKOS_SHARED_H_
