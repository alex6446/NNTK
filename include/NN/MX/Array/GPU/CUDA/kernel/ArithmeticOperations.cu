#include "NN/Core/Base.hpp"

namespace NN
{
namespace MX
{
namespace internal
{

#define MX_ARRAY_OPERATION_AA_CUDA_KERNEL(op, name) \
    template<typename T> \
    __global__ \
    void \
    name##_array_array_cuda_kernel(T *result, const T *array, size_t size) \
    { \
        int index = blockIdx.x * blockDim.x + threadIdx.x; \
        int stride = blockDim.x * gridDim.x; \
        for (size_t i = index; i < size; i+=stride) { \
            result[i] op##= array[i]; \
        } \
    }
        //int index = threadIdx.x; \
        //int stride = blockDim.x; \

    MX_ARRAY_OPERATION_AA_CUDA_KERNEL(+, add)
    MX_ARRAY_OPERATION_AA_CUDA_KERNEL(-, subtract)
    MX_ARRAY_OPERATION_AA_CUDA_KERNEL(*, multiply)
    MX_ARRAY_OPERATION_AA_CUDA_KERNEL(/, divide)


} // namespace NN
} // namespace MX
} // namespace internal
