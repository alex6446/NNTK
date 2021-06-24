#pragma once

#include "NNTK/Core/Device.hpp"
#include "NNTK/Core/Error.hpp"

#include "../Predefinition.hpp"

#include "CUDA/ArithmeticOperations.cu"

namespace NN::MX::internal
{

#define MX_ARRAY_OPERATION_AA_GPU(name, check)                                \
  template<typename T, Device D, Device V>                                    \
    void                                                                      \
    name##_array_array_gpu(Array<T, D> &result, const Array<T, V> &array)     \
    {                                                                         \
      if (check)                                                              \
        for (auto &e : array)                                                 \
          NN_RUNTIME_ERROR(e == 0, "array: division by zero")                 \
      name##_array_array_cuda(result.data(), array.data(), result.size());    \
    }

  MX_ARRAY_OPERATION_AA_GPU(add, 0)
  MX_ARRAY_OPERATION_AA_GPU(subtract, 0)
  MX_ARRAY_OPERATION_AA_GPU(multiply, 0)
  MX_ARRAY_OPERATION_AA_GPU(divide, 1)

#define MX_ARRAY_OPERATION_AV_GPU(name)                                       \
  template<typename T, Device D>                                              \
    void                                                                      \
    name##_array_value_gpu(Array<T, D> &result, const T &value)               \
    {                                                                         \
      name##_array_value_cuda(result.data(), value, result.size());           \
    }

  MX_ARRAY_OPERATION_AV_GPU(add)
  MX_ARRAY_OPERATION_AV_GPU(subtract)
  MX_ARRAY_OPERATION_AV_GPU(multiply)
  MX_ARRAY_OPERATION_AV_GPU(divide)

#define MX_ARRAY_OPERATION_VA_GPU(op, name, check)                            \
  template<typename T, Device D>                                              \
    void                                                                      \
    name##_value_array_gpu(const T & value, Array<T, D> &result)              \
    {                                                                         \
      if (check)                                                              \
        for (auto &e : result)                                                \
          NN_RUNTIME_ERROR(e == 0, "division by zero")                        \
      name##_value_array_cuda(value, result.data(), result.size());           \
    }

  MX_ARRAY_OPERATION_VA_GPU(-, subtract, 0)
  MX_ARRAY_OPERATION_VA_GPU(/, divide, 1)


} // namespace NN::MX::internal

