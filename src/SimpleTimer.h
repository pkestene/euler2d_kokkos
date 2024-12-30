// SPDX-FileCopyrightText: 2024 euler2d_kokkos authors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * \file SimpleTimer.h
 * \brief A simple timer class.
 *
 * \author Pierre Kestener
 * \date 29 Oct 2010
 *
 */
#ifndef SIMPLE_TIMER_H_
#define SIMPLE_TIMER_H_

#include <time.h>
#include <sys/time.h> // for gettimeofday and struct timeval

typedef struct timeval timeval_t;

/**
 * \brief a simple Timer class.
 * If MPI is enabled, should we use MPI_WTime instead of gettimeofday (?!?)
 */
class SimpleTimer
{
public:
  /** default constructor, timing starts rightaway */
  SimpleTimer();

  SimpleTimer(double t);
  SimpleTimer(SimpleTimer const & aTimer);
  virtual ~SimpleTimer();

  /** start time measure */
  virtual void
  start();

  /** stop time measure and add result to total_time */
  virtual void
  stop();

  /** return elapsed time in seconds (as stored in total_time) */
  virtual double
  elapsed() const;

protected:
  double start_time;

  /** store total accumulated timings */
  double total_time;

}; // class SimpleTimer


#endif // SIMPLE_TIMER_H_
