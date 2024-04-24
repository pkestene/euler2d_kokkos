/**
 * \file cnpy_io.h
 * \brief converting numpy files to/from a Kokkos::View
 */
#ifndef EULER2D_CNPY_IO_H_
#define EULER2D_CNPY_IO_H_

#include "../kokkos_shared.h"

#include <vector>
#include <sstream>
#include <string>
#include <iomanip> // for std::setfill

#include "./cnpy.h"

namespace euler2d
{

// =============================================================================
// =============================================================================
/**
 * Save a multidimensional Kokkos::View into a file using numpy format.
 *
 * example of use in python:
 *   data = np.load('data.npy')
 *
 * \tparam View is a Kokkos::View class, we check that the view is accessible from host
 *
 * \param[in] v a View instance containing a data array to be saved
 */
template <class View>
void
save_cnpy(View v, std::string name = "")
{
  constexpr bool view_accessible_from_host =
    Kokkos::SpaceAccessibility</*AccessSpace=*/Kokkos::HostSpace,
                               /*MemorySpace=*/typename View::memory_space>::accessible;

  if constexpr (view_accessible_from_host)
  {

    // save in numpy format
    std::vector<size_t> shape;
    if constexpr (View::rank >= 1)
      shape.push_back(v.extent(0));
    if constexpr (View::rank >= 2)
      shape.push_back(v.extent(1));
    if constexpr (View::rank >= 3)
      shape.push_back(v.extent(2));

    std::ostringstream ss;
    if (name.size() > 0)
      ss << name << ".npy";
    else
      ss << v.label() << ".npy";
    cnpy::npy_save(ss.str(), v.data(), shape, "w");
  }
} // save_cnpy

} // namespace euler2d

#endif // EULER2D_CNPY_IO_H_
