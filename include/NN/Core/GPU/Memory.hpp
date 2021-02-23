#include "CUDA/Memory.hpp"

namespace NN
{
namespace internal
{

    template<typename T>
    void
    allocate_memory_gpu(T **pointer, size_t size)
    {
        //*pointer = new T[size];
        allocate_memory_cuda((void **)pointer, sizeof(T) * size);
    }

    template<typename T>
    void
    free_memory_gpu(T **pointer)
    {
        //delete[] *pointer;
        free_memory_cuda((void **)pointer);
    }

} // namespace NN
} // namespace internal