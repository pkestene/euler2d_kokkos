// SPDX-FileCopyrightText: 2024 euler2d_kokkos authors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * \file ComputeRadialProfile.h
 */
#ifndef EULER2D_COMPUTE_RADIAL_PROFILE_FUNCTOR_H_
#define EULER2D_COMPUTE_RADIAL_PROFILE_FUNCTOR_H_

#include "kokkos_shared.h"
#include "HydroBaseFunctor.h"
#include "HydroParams.h"
#include "real_type.h"

// other utilities
#include "cnpy/cnpy_io.h"

namespace euler2d
{

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Compute radial profile by averaging all direction.
 *
 * This functor is mainly aimed at checking numerical and analytical solution of the radial Sedov
 * blast.
 *
 * \tparam device_t is the Kokkos device use for computation (CPU, GPU, ...)
 */
template <typename device_t>
class ComputeRadialProfileFunctor : HydroBaseFunctor
{

public:
  using DataArray_t = DataArray<device_t>;

  //! our kokkos execution space
  using exec_space = typename device_t::execution_space;

private:
  //! heavy data - conservative variables
  DataArray_t m_Udata;

  //! number of bins in radial direction
  const int m_nbins;

  //! max radial distance
  const real_t m_max_radial_distance;

  //! center of the box
  Kokkos::Array<real_t, 2> m_box_center;

  //! radial distance histogram
  Kokkos::View<int *, device_t, Kokkos::MemoryTraits<Kokkos::Atomic>> m_radial_distance_histo;

  //! averaged density radial profile
  Kokkos::View<real_t *, device_t, Kokkos::MemoryTraits<Kokkos::Atomic>> m_density_profile;

public:
  ComputeRadialProfileFunctor(HydroParams              params,
                              DataArray_t              Udata,
                              real_t                   max_radial_distance,
                              Kokkos::Array<real_t, 2> box_center)
    : HydroBaseFunctor(params)
    , m_Udata(Udata)
    , m_nbins(params.blast_nbins)
    , m_max_radial_distance(max_radial_distance)
    , m_box_center(box_center)
    , m_radial_distance_histo("radial_distance_histo", m_nbins)
    , m_density_profile("density_profile", m_nbins)
  {
    Kokkos::deep_copy(m_radial_distance_histo, 0);
    Kokkos::deep_copy(m_density_profile, ZERO_F);
  };

  // ====================================================================
  // ====================================================================
  //! static method which does it all: create and execute functor using range policy
  //!
  static void
  apply(HydroParams params, DataArray_t Udata)
  {

    const real_t xmin = params.xmin;
    const real_t ymin = params.ymin;
    const real_t xmax = params.xmax;
    const real_t ymax = params.ymax;

    // get center of the box coordinates
    Kokkos::Array<real_t, 2> box_center;

    box_center[IX] = (xmin + xmax) / 2;
    box_center[IY] = (ymin + ymax) / 2;

    // compute maximum radial distance
    const auto Dx = (xmax - xmin) / 2;
    const auto Dy = (ymax - ymin) / 2;
    const auto max_radial_distance = sqrt(Dx * Dx + Dy * Dy);

    ComputeRadialProfileFunctor functor(params, Udata, max_radial_distance, box_center);

    Kokkos::parallel_for(
      "euler2d::ComputeRadialProfileFunctor",
      Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>({ 0, 0 }, { params.isize, params.jsize }),
      functor);

    // once the radial profile is computed, gather all the MPI pieces
    const auto nbins = params.blast_nbins;
    auto       total_radial_distance_histo =
      Kokkos::View<int *, device_t>("total_radial_distance_histo", nbins);

    const auto total_density_profile =
      Kokkos::View<real_t *, device_t>("total_density_profile", nbins);

    // then we compute the profile and output in numpy format
    auto rad_dist =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, functor.m_radial_distance_histo);
    auto dens_prof =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, functor.m_density_profile);

    auto       distances = Kokkos::View<real_t *, Kokkos::HostSpace>("distances", nbins);
    const auto dr = max_radial_distance / nbins;

    for (int i = 0; i < nbins; ++i)
    {
      distances(i) = (i + 0.5) * dr;
      dens_prof(i) /= rad_dist(i);
    }

    // now we can output results
    save_cnpy(distances, "sedov_blast_radial_distances");
    save_cnpy(dens_prof, "sedov_blast_density_profile");

  } // apply

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i, const int & j) const
  {

    const int    ghostWidth = params.ghostWidth;
    const real_t xmin = params.xmin;
    const real_t ymin = params.ymin;
    const real_t dx = params.dx;
    const real_t dy = params.dy;
    const real_t x = xmin + dx / 2 + (i - ghostWidth) * dx;
    const real_t y = ymin + dy / 2 + (j - ghostWidth) * dy;

    const auto distance = sqrt((x - m_box_center[IX]) * (x - m_box_center[IX]) +
                               (y - m_box_center[IY]) * (y - m_box_center[IY]));

    // compute bin number from distance to center of the box
    const auto bin = (int)(distance / m_max_radial_distance * m_nbins);

    // update density profile
    m_radial_distance_histo(bin) += 1;

    m_density_profile(bin) += m_Udata(i, j, ID);

  } // operator()

}; // ComputeRadialProfileFunctor

} // namespace euler2d

#endif // EULER2D_COMPUTE_RADIAL_PROFILE_FUNCTOR_H_
