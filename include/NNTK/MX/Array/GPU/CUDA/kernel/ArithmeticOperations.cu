#pragma once

#include <iostream>
#include "NNTK/Core/Error.hpp"

namespace NN::MX::internal
{

#define MX_ARRAY_OPERATION_AA_CUDA_KERNEL(op, name)                           \
  template<typename T>                                                        \
    __global__ void                                                           \
    name##_array_array_cuda_kernel(T* result, const T* array, size_t size)    \
    {                                                                         \
      int index = blockIdx.x * blockDim.x + threadIdx.x;                      \
      int stride = blockDim.x * gridDim.x;                                    \
      for (size_t i = index; i < size; i+=stride)                             \
        result[i] op##= array[i];                                             \
    }

  MX_ARRAY_OPERATION_AA_CUDA_KERNEL(+, add)
  MX_ARRAY_OPERATION_AA_CUDA_KERNEL(*, multiply)
  MX_ARRAY_OPERATION_AA_CUDA_KERNEL(-, subtract)
  MX_ARRAY_OPERATION_AA_CUDA_KERNEL(/, divide)

#define MX_ARRAY_OPERATION_AA_REV_CUDA_KERNEL(op, name)                       \
  template<typename T>                                                        \
    __global__ void                                                           \
    name##_array_array_reverse_cuda_kernel(T* result,                         \
                                           const T* array, size_t size)       \
    {                                                                         \
      int index = blockIdx.x * blockDim.x + threadIdx.x;                      \
      int stride = blockDim.x * gridDim.x;                                    \
      for (size_t i = index; i < size; i+=stride)                             \
        result[i] = array[i] op result[i];                                    \
    }

  MX_ARRAY_OPERATION_AA_REV_CUDA_KERNEL(-, subtract)
  MX_ARRAY_OPERATION_AA_REV_CUDA_KERNEL(/, divide)

#define MX_ARRAY_OPERATION_AV_CUDA_KERNEL(op, name)                           \
  template<typename T>                                                        \
    __global__ void                                                           \
    name##_array_value_cuda_kernel(T* result, T value, size_t size)           \
    {                                                                         \
      int index = blockIdx.x * blockDim.x + threadIdx.x;                      \
      int stride = blockDim.x * gridDim.x;                                    \
      for (size_t i = index; i < size; i+=stride)                             \
        result[i] op##= value;                                                \
    }

  MX_ARRAY_OPERATION_AV_CUDA_KERNEL(+, add)
  MX_ARRAY_OPERATION_AV_CUDA_KERNEL(*, multiply)
  MX_ARRAY_OPERATION_AV_CUDA_KERNEL(-, subtract)
  MX_ARRAY_OPERATION_AV_CUDA_KERNEL(/, divide)

#define MX_ARRAY_OPERATION_VA_CUDA_KERNEL(op, name)                           \
  template<typename T>                                                        \
    __global__ void                                                           \
    name##_value_array_cuda_kernel(T value, T* result, size_t size)           \
    {                                                                         \
      int index = blockIdx.x * blockDim.x + threadIdx.x;                      \
      int stride = blockDim.x * gridDim.x;                                    \
      for (size_t i = index; i < size; i+=stride)                             \
        result[i] = value op result[i];                                       \
    }

  MX_ARRAY_OPERATION_VA_CUDA_KERNEL(-, subtract)
  MX_ARRAY_OPERATION_VA_CUDA_KERNEL(/, divide)


} // namespace NN::MX::internal
