/**
 * \file Timer.cpp
 * \brief a simpe Timer class implementation.
 * 
 * \author Pierre Kestener
 * \date 29 Oct 2010
 *
 */

#include "Timer.h"

#include <stdexcept>

////////////////////////////////////////////////////////////////////////////////
// Timer class methods body
////////////////////////////////////////////////////////////////////////////////

// =======================================================
// =======================================================
Timer::Timer() {
  start_time = 0.0;
  total_time = 0.0;
  start();
} // Timer::Timer

// =======================================================
// =======================================================
Timer::Timer(double t) 
{
    
  //start_time.tv_sec = time_t(t);
  //start_time.tv_usec = (t - start_time.tv_sec) * 1e6;
  // start_time.tv_sec = 0;
  // start_time.tv_usec = 0;
  start_time = 0;
  total_time = t;
    
} // Timer::Timer

  // =======================================================
  // =======================================================
Timer::Timer(Timer const& aTimer) : start_time(aTimer.start_time), total_time(aTimer.total_time)
{
} // Timer::Timer

  // =======================================================
  // =======================================================
Timer::~Timer()
{
} // Timer::~Timer

  // =======================================================
  // =======================================================
void Timer::start() 
{

  timeval_t now;
  if (-1 == gettimeofday(&now, 0))
    throw std::runtime_error("Timer: Couldn't initialize start_time time");

  start_time = double(now.tv_sec) +  (double(now.tv_usec) * 1e-6);
  
} // Timer::start
  
  // =======================================================
  // =======================================================
void Timer::stop()
{
  double now_d;
  timeval_t now;
  if (-1 == gettimeofday(&now, 0))
    throw std::runtime_error("Couldn't get current time");
    
  now_d = double(now.tv_sec) + (double(now.tv_usec) * 1e-6);
  
  total_time += (now_d-start_time);

} // Timer::stop

  // =======================================================
  // =======================================================
double Timer::elapsed() const
{

  return total_time;

} // Timer::elapsed
