//
// Created by developer on 7/19/20.
//

#ifndef CS535_GRAPHSEARCH_ERROR_CHECKER_H
#define CS535_GRAPHSEARCH_ERROR_CHECKER_H

#include <cuda_runtime_api.h>
#include <iostream>

inline void check(cudaError_t result, char const* const func, const char* const file, int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
        file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
    cudaDeviceReset();
    // Make sure we call CUDA Device Reset before exiting
    exit(EXIT_FAILURE);
  }
}

#ifndef CudaCatchError
#define CudaCatchError(val) ::check((val), #val, __FILE__, __LINE__)
#endif


#endif //CS535_GRAPHSEARCH_ERROR_CHECKER_H
