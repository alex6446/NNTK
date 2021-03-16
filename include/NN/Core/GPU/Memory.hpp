#pragma once

#include "CUDA/Memory.hpp"
#include <iostream>
#include <ostream>

namespace NN
{
namespace internal
{

template<typename T>
void
allocate_memory_gpu(T **pointer, size_t size)
{
    //*pointer = new T[size];
    std::cout << sizeof(T) * size << std::endl;
    allocate_memory_cuda((void **)pointer, sizeof(T) * size);
}

template<typename T>
void
free_memory_gpu(T **pointer)
{
    //delete[] *pointer;
    free_memory_cuda((void **)pointer);
}

} // namespace internal

} // namespace NN
