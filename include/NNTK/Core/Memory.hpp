#pragma once

#include <cstddef>

#ifdef NN_GPU_ENABLED
#include "GPU/Memory.hpp"
#endif

namespace NN::internal
{

template<typename T>
void
allocate_memory_cpu(T **pointer, size_t size)
{ *pointer = new T[size]; }

template<typename T>
void
free_memory_cpu(T **pointer)
{ delete[] *pointer; }

void copy_memory_cpu(void *dst, const void *src, size_t count);

} // namespace NN::internal
