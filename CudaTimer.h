/**
 * \file CudaTimer.h
 * \brief A simple timer class for CUDA based on events.
 *
 * \author Pierre Kestener
 * \date 30 Oct 2010
 *
 */
#ifndef CUDA_TIMER_H_
#define CUDA_TIMER_H_

#include <cuda_runtime.h>

/**
 * \brief a simple timer for CUDA kernel.
 */
class CudaTimer
{
protected:
  cudaEvent_t startEv, stopEv;
  double total_time;
  
public:
  CudaTimer() {
    cudaEventCreate(&startEv);
    cudaEventCreate(&stopEv);
    total_time = 0.0;
  }
  
  ~CudaTimer() {
    cudaEventDestroy(startEv);
    cudaEventDestroy(stopEv);
  }
  
  void start() {
    cudaEventRecord(startEv, 0);
  }
  
  void reset() {
    total_time = 0.0;
  }
  
  /** stop timer and accumulate time in seconds */
  void stop() {
    float gpuTime;
    cudaEventRecord(stopEv, 0);
    cudaEventSynchronize(stopEv);
    cudaEventElapsedTime(&gpuTime, startEv, stopEv);
    total_time += (double)1e-3*gpuTime;
  }
  
  /** return elapsed time in seconds (as record in total_time) */
  double elapsed() const {
    return total_time;
  }
  
}; // class CudaTimer
  
#endif // CUDA_TIMER_H_
