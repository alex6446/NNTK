#pragma once

#include <cstddef>

namespace NN::internal
{

void allocate_memory_cuda(void **pointer, size_t bytes);
void free_memory_cuda(void **pointer);
void copy_memory_cuda(void *dst, const void *src, size_t count);

} // namespace NN::internal
