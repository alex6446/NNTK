#include "NN/Core/GPU/CUDA/Memory.hpp"

namespace NN
{
namespace internal
{

    void
    allocate_memory_cuda(void **pointer, size_t bytes)
    {
        cudaMallocManaged(pointer, bytes);
    }

    void
    free_memory_cuda(void **pointer)
    {
        cudaFree(pointer);
    }

} // namespace NN
} // namespace internal
