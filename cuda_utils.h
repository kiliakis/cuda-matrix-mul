#ifndef __CUDA_UTILS_H__
#define __CUDA_UTILS_H__

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

void check_cuda(cudaError_t result, char const *const func,
                const char *const file, int const line) {
  if (result) {
    printf("CUDA error at %s: %d code = %d (%s) %s", file, line,
           static_cast<unsigned int>(result), cudaGetErrorName(result), func);
    exit(-1);
  }
}

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)


#endif