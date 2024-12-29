/**
 * \file SimpleTimer.cpp
 * \brief a simpe Timer class implementation.
 *
 * \author Pierre Kestener
 * \date 29 Oct 2010
 *
 */

#include "SimpleTimer.h"

#include <stdexcept>

////////////////////////////////////////////////////////////////////////////////
// SimpleTimer class methods body
////////////////////////////////////////////////////////////////////////////////

// =======================================================
// =======================================================
SimpleTimer::SimpleTimer()
{
  start_time = 0.0;
  total_time = 0.0;
  start();
} // SimpleTimer::SimpleTimer

// =======================================================
// =======================================================
SimpleTimer::SimpleTimer(double t)
{

  // start_time.tv_sec = time_t(t);
  // start_time.tv_usec = (t - start_time.tv_sec) * 1e6;
  //  start_time.tv_sec = 0;
  //  start_time.tv_usec = 0;
  start_time = 0;
  total_time = t;

} // SimpleTimer::SimpleTimer

// =======================================================
// =======================================================
SimpleTimer::SimpleTimer(SimpleTimer const & aTimer)
  : start_time(aTimer.start_time)
  , total_time(aTimer.total_time)
{} // SimpleTimer::SimpleTimer

// =======================================================
// =======================================================
SimpleTimer::~SimpleTimer() {} // SimpleTimer::~SimpleTimer

// =======================================================
// =======================================================
void
SimpleTimer::start()
{

  timeval_t now;
  if (-1 == gettimeofday(&now, 0))
    throw std::runtime_error("SimpleTimer: Couldn't initialize start_time time");

  start_time = double(now.tv_sec) + (double(now.tv_usec) * 1e-6);

} // SimpleTimer::start

// =======================================================
// =======================================================
void
SimpleTimer::stop()
{
  double    now_d;
  timeval_t now;
  if (-1 == gettimeofday(&now, 0))
    throw std::runtime_error("Couldn't get current time");

  now_d = double(now.tv_sec) + (double(now.tv_usec) * 1e-6);

  total_time += (now_d - start_time);

} // SimpleTimer::stop

// =======================================================
// =======================================================
double
SimpleTimer::elapsed() const
{

  return total_time;

} // SimpleTimer::elapsed
