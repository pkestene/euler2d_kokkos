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
using MyArray = Kokkos::View<real_t*, Device>;
using MyArrayHost = MyArray::HostMirror;

using DataArray = Kokkos::Array<MyArray, NBVAR>;
using DataArrayHost = Kokkos::Array<MyArrayHost, NBVAR>;

/// a POD data structure to store local conservative / primitive variables
using HydroState = Kokkos::Array<real_t,NBVAR>;

DataArray allocate_DataArray(std::string name, int size);
DataArrayHost create_mirror(DataArray data);

template<class array1, class array2>
void deep_copy(array1 dest, array2 src)
{

  for (int i=0; i<NBVAR; ++i)
    Kokkos::deep_copy(dest[i], src[i]);

} // deep_copy

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
