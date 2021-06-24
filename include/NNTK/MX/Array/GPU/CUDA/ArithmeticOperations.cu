#pragma once

#include <cstddef>
#include <iostream>

#include "NNTK/Core/Error.hpp"

#include "kernel/ArithmeticOperations.cu"

namespace NN::MX::internal
{

#define BLOCK_SIZE 256

#define MX_ARRAY_OPERATION_AA_CUDA(name)                                      \
  template<typename T>                                                        \
    void                                                                      \
    name##_array_array_cuda(T* d_result, const T* d_array, size_t size)       \
    {                                                                         \
      name##_array_array_cuda_kernel                                          \
        <<<size / BLOCK_SIZE, BLOCK_SIZE>>>                                   \
          (d_result, d_array, size);                                          \
    }

  MX_ARRAY_OPERATION_AA_CUDA(add)
  MX_ARRAY_OPERATION_AA_CUDA(multiply)
  MX_ARRAY_OPERATION_AA_CUDA(subtract)
  MX_ARRAY_OPERATION_AA_CUDA(divide)

#define MX_ARRAY_OPERATION_AA_REV_CUDA(name)                                  \
  template<typename T>                                                        \
    void                                                                      \
    name##_array_array_reverse_cuda(T* d_result, const T* d_array,            \
                                    size_t size)                              \
    {                                                                         \
      name##_array_array_reverse_cuda_kernel                                  \
        <<<size / BLOCK_SIZE, BLOCK_SIZE>>>                                   \
          (d_result, d_array, size);                                          \
    }

  MX_ARRAY_OPERATION_AA_REV_CUDA(subtract)
  MX_ARRAY_OPERATION_AA_REV_CUDA(divide)

#define MX_ARRAY_OPERATION_AV_CUDA(name)                                      \
  template<typename T>                                                        \
    void                                                                      \
    name##_array_value_cuda(T* d_result, const T& value, size_t size)         \
    {                                                                         \
      name##_array_value_cuda_kernel                                          \
        <<<size / BLOCK_SIZE, BLOCK_SIZE>>>                                   \
          (d_result, value, size);                                            \
    }

  MX_ARRAY_OPERATION_AV_CUDA(add)
  MX_ARRAY_OPERATION_AV_CUDA(multiply)
  MX_ARRAY_OPERATION_AV_CUDA(subtract)
  MX_ARRAY_OPERATION_AV_CUDA(divide)

#define MX_ARRAY_OPERATION_VA_CUDA(name)                                      \
  template<typename T>                                                        \
    void                                                                      \
    name##_value_array_cuda(const T& value, T* d_result, size_t size)         \
    {                                                                         \
      name##_value_array_cuda_kernel                                          \
        <<<size / BLOCK_SIZE, BLOCK_SIZE>>>                                   \
          (value, d_result, size);                                            \
    }

  MX_ARRAY_OPERATION_VA_CUDA(subtract)
  MX_ARRAY_OPERATION_VA_CUDA(divide)


} // namespace NN::MX::internal

