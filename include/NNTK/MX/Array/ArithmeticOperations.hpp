#pragma once

#include "NNTK/Core/Device.hpp"
#include "NNTK/Core/Error.hpp"

#include "Predefinition.hpp"

#ifdef NN_GPU_ENABLED
#include "GPU/ArithmeticOperations.hpp"
#endif

namespace NN::MX::internal
{

#define MX_ARRAY_OPERATION_AA_CPU(op, name, check)                            \
  template<typename T, Device D, Device V>                                    \
    void                                                                      \
    name##_array_array_cpu(Array<T, D> &result, const Array<T, V> &array)     \
    {                                                                         \
      for (T *resd = result.data(), *arrd = (T*) array.data();                \
           resd != result.data() + result.size(); ++resd, ++arrd) {           \
        if (check) NN_RUNTIME_ERROR(!(*arrd), "array: division by zero")      \
        *resd op##= *arrd;                                                    \
      }                                                                       \
    }

  MX_ARRAY_OPERATION_AA_CPU(+, add, 0)
  MX_ARRAY_OPERATION_AA_CPU(-, subtract, 0)
  MX_ARRAY_OPERATION_AA_CPU(*, multiply, 0)
  MX_ARRAY_OPERATION_AA_CPU(/, divide, 1)

#define MX_ARRAY_OPERATION_AV_CPU(op, name)                                   \
  template<typename T, Device D>                                              \
    void                                                                      \
    name##_array_value_cpu(Array<T, D> &result, const T &value)               \
    {                                                                         \
      for (T *e = result.data(); e != result.data() + result.size(); ++e)     \
        *e op##= value;                                                       \
    }

  MX_ARRAY_OPERATION_AV_CPU(+, add)
  MX_ARRAY_OPERATION_AV_CPU(-, subtract)
  MX_ARRAY_OPERATION_AV_CPU(*, multiply)
  MX_ARRAY_OPERATION_AV_CPU(/, divide)

#define MX_ARRAY_OPERATION_VA_CPU(op, name, check)                            \
  template<typename T, Device D>                                              \
    void                                                                      \
    name##_value_array_cpu(const T &value, Array<T, D> &result)               \
    {                                                                         \
      for (T *e = result.data(); e != result.data() + result.size(); ++e) {   \
        if (check) NN_RUNTIME_ERROR(*e == 0, "division by zero")              \
        *e = value op *e;                                                     \
      }                                                                       \
    }

  MX_ARRAY_OPERATION_VA_CPU(-, subtract, 0)
  MX_ARRAY_OPERATION_VA_CPU(/, divide, 1)

} // namespace NN::MX::internal
