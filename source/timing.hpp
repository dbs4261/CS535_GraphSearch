//
// Created by Daniel Simon on 8/9/20.
//

#ifndef CS535_GRAPHSEARCH_TIMING_H
#define CS535_GRAPHSEARCH_TIMING_H

#include <chrono>
#include <type_traits>
#include <ostream>

#include <cuda_runtime_api.h>

#include "source/error_checker.h"

struct RaiiCudaTimer {
 protected:
  cudaEvent_t start_event;
  cudaEvent_t end_event;
  cudaStream_t stream;
  std::chrono::duration<float, std::milli>& elapsed;
 public:
  RaiiCudaTimer(std::chrono::duration<float, std::milli>& elapsed_target, cudaStream_t stream=nullptr) : elapsed(elapsed_target), stream(stream) {
    CudaCatchError(cudaEventCreate(&start_event));
    CudaCatchError(cudaEventCreate(&end_event));
    CudaCatchError(cudaEventRecord(start_event, stream));
  }
  ~RaiiCudaTimer() {
    CudaCatchError(cudaEventRecord(end_event, stream));
    CudaCatchError(cudaStreamSynchronize(stream));
    float temp;
    CudaCatchError(cudaEventElapsedTime(&temp, start_event, end_event));
    elapsed = std::chrono::duration<float, std::milli>(temp);
    CudaCatchError(cudaEventDestroy(start_event));
    CudaCatchError(cudaEventDestroy(end_event));
  }
};

struct RaiiTimer {
 protected:
  std::chrono::high_resolution_clock::time_point start;
  std::chrono::duration<float, std::milli>& elapsed;
 public:
  RaiiTimer() = delete;
  RaiiTimer(std::chrono::duration<float, std::milli>& elapsed_target)
      : elapsed(elapsed_target), start(std::chrono::high_resolution_clock::now()) {}
  ~RaiiTimer() {
    elapsed = std::chrono::high_resolution_clock::now() - start;
  }
};

struct TimingWrapper {
  std::chrono::duration<float, std::milli> elapsed;

  template <typename Function, typename ... Args>
  std::invoke_result_t<Function, Args...> Run(Function function, Args&& ... args) {
    RaiiCudaTimer timer(elapsed);
    return function(std::forward<Args>(args)...);
  }

  friend std::ostream& operator<<(std::ostream& os, const TimingWrapper& timer) {
    os << timer.elapsed.count() << "ms";
    return os;
  }
};

struct CudaTimers {
  TimingWrapper upload;
  TimingWrapper execution;
  TimingWrapper download;

  friend std::ostream& operator<<(std::ostream& os, const CudaTimers& timer) {
    std::chrono::duration<float, std::milli> total = timer.upload.elapsed + timer.execution.elapsed + timer.download.elapsed;
    os << "Total: " << total.count() << "ms [Upload: " << timer.upload << " Execution: "
        << timer.execution << " Download: " << timer.download << "]";
    return os;
  }
};

#endif //CS535_GRAPHSEARCH_TIMING_H
