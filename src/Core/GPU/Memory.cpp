#include "NNTK/Core/GPU/CUDA/Memory.hpp"

namespace NN::internal
{

void
copy_memory_gpu(void *dst, const void *src, std::size_t count)
{ copy_memory_cuda(dst, src, count); }

} // namespace NN::internal

