#include "NNTK/Core/GPU/CUDA/Memory.hpp"

namespace NN
{
namespace internal
{

void
allocate_memory_cuda(void **pointer, size_t bytes)
{ cudaMallocManaged(pointer, bytes); }

void
free_memory_cuda(void **pointer)
{ cudaFree(pointer); }

void
copy_memory_cuda(void *dst, const void *src, std::size_t count)
{ cudaMemcpy(dst, src, count, cudaMemcpyDefault); }

} // namespace internal

} // namespace NN
