#pragma once

#include <iostream>
#include <ostream>

#include "CUDA/Memory.hpp"

namespace NN::internal
{

template<typename T>
void
allocate_memory_gpu(T **pointer, size_t size)
{ allocate_memory_cuda((void **)pointer, sizeof(T) * size); }

template<typename T>
void
free_memory_gpu(T **pointer)
{ free_memory_cuda((void **)pointer); }

void copy_memory_gpu(void *dst, const void *src, size_t count);

} // namespace NN::internal
