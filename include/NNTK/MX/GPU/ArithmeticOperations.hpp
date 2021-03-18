#pragma once

#include "NNTK/Core/Error.hpp"

#include "CUDA/ArithmeticOperations.cu"

namespace NN
{
namespace MX
{

template<typename T>
class Array;

namespace internal
{

#define MX_ARRAY_OPERATION_AA_GPU(name, check) \
template<typename T> \
void \
name##_array_array_gpu(Array<T> &result, const Array<T> &array) \
{ \
    if (check) { \
        for (auto &i : array) \
            NN_RUNTIME_ERROR(!i, "array: division by zero") \
    } \
    name##_array_array_cuda(result.data(), array.data(), result.size()); \
}

MX_ARRAY_OPERATION_AA_GPU(add, 0)
MX_ARRAY_OPERATION_AA_GPU(subtract, 0)
MX_ARRAY_OPERATION_AA_GPU(multiply, 0)
MX_ARRAY_OPERATION_AA_GPU(divide, 1)


} // namespace internal

} // namespace MX

} // namespace NN

