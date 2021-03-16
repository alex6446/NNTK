#pragma once

#include "NN/Core/Error.hpp"

namespace NN
{
namespace MX
{

template<typename T>
class Array;

namespace internal
{

template<typename T>
using size_type = typename Array<T>::size_type;

#define MX_ARRAY_OPERATION_AA_CPU(op, name, check) \
template<typename T> \
void \
name##_array_array_cpu(Array<T> &result, const Array<T> &array) \
{ \
    for (size_type<T> i = 0; i < result.size(); ++i) { \
        if (check) { NN_RUNTIME_ERROR(!array.data(i), "array: division by zero") } \
        result.data(i) op##= array.data(i); \
    } \
}

MX_ARRAY_OPERATION_AA_CPU(+, add, 0)
MX_ARRAY_OPERATION_AA_CPU(-, subtract, 0)
MX_ARRAY_OPERATION_AA_CPU(*, multiply, 0)
MX_ARRAY_OPERATION_AA_CPU(/, divide, 1)


} // namespace internal

} // namespace MX

} // namespace NN
