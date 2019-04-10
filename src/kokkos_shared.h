#ifndef KOKKOS_SHARED_H_
#define KOKKOS_SHARED_H_

#include <Kokkos_Core.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>

#include "real_type.h"
#include "hydro_common.h"

using Device = Kokkos::DefaultExecutionSpace;

/**
 * Main data array type alias (based on Kokkos::View).
 *
 * index linearization (layout left)
 * index = i + isize*j
 */
typedef Kokkos::View<real_t*[NBVAR], Kokkos::LayoutLeft, Device> DataArray;
typedef DataArray::HostMirror DataArrayHost;

/// a POD data structure to store local conservative / primitive variables
using HydroState = Kokkos::Array<real_t,NBVAR>;

/**
 * Retrieve cartesian coordinates from index, using memory layout left.
 *
 */
KOKKOS_INLINE_FUNCTION
void index2coord(int index, int &i, int &j, int Nx, int Ny) {
  j = index / Nx;
  i = index - j*Nx;
}

KOKKOS_INLINE_FUNCTION
int coord2index(int i, int j, int Nx, int Ny) {
  return i + Nx*j; // left layout
}

#endif // KOKKOS_SHARED_H_
