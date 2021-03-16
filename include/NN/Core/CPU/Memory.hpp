#pragma once

#include <cstddef>

namespace NN
{
namespace internal
{

    template<typename T>
    void
    allocate_memory_cpu(T **pointer, size_t size)
    {
        *pointer = new T[size];
    }

    template<typename T>
    void
    free_memory_cpu(T **pointer)
    {
        delete[] *pointer;
    }

} // namespace internal

} // namespace NN
