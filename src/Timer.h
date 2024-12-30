// SPDX-FileCopyrightText: 2024 euler2d_kokkos authors
//
// SPDX-License-Identifier: Apache-2.0

#ifndef EULER_KOKKOS_TIMER_H_
#define EULER_KOKKOS_TIMER_H_

#include <Kokkos_Macros.hpp> // for KOKKOS_ENABLE_XXX

#ifdef KOKKOS_ENABLE_CUDA
#  include "CudaTimer.h"
using Timer = CudaTimer;
#else
#  include "SimpleTimer.h"
using Timer = SimpleTimer;
#endif

#endif // EULER_KOKKOS_TIMER_H_
