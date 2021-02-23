#include <cstddef>
#include <iostream>

#include "NN/Core/Base.hpp"

#include "kernel/ArithmeticOperations.cu"

namespace NN
{
namespace MX
{
namespace internal
{

using namespace std;
#define BLOCK_SIZE 256

#define MX_ARRAY_OPERATION_AA_CUDA(name, check) \
    template<typename T> \
    void \
    name##_array_array_cuda(T *d_result, const T *d_array, size_t size) \
    { \
        std::cout << "Running on GPU!" << std::endl; \
        size_t bytes = size * sizeof(T); \
        int device = -1; \
        cudaGetDevice(&device); \
        cudaMemPrefetchAsync(d_result, bytes, device, NULL); \
        cudaMemPrefetchAsync(d_array, bytes, device, NULL); \
        for (int i = 0; i < 3; ++i) { \
        clock_t begin = clock(); \
        name##_array_array_cuda_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_result, d_array, size); \
        cudaDeviceSynchronize(); \
        clock_t end = clock(); \
        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC; \
        cout << "TIME: " << elapsed_secs << "s" << endl; \
        }\
    }
        //std::cout << "Running on GPU!" << std::endl; \
        //T *d_result, *d_array; \
        //size_t bytes = size * sizeof(T); \
        //\
        //cudaMalloc(&d_result, bytes); \
        //cudaMalloc(&d_array, bytes); \
        //\
        //cudaMemcpy(d_result, h_result, bytes, cudaMemcpyHostToDevice); \
        //cudaMemcpy(d_array, h_array, bytes, cudaMemcpyHostToDevice); \
        //\
        //using namespace std; \
        //clock_t begin = clock(); \
        //name##_array_array_cuda_kernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_result, d_array, size); \
        //cudaDeviceSynchronize(); \
        //clock_t end = clock(); \
        //double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC; \
        //\
        //cudaMemcpy(h_result, d_result, bytes, cudaMemcpyDeviceToHost);\
        //cudaFree(d_result); \
        //cudaFree(d_array); \
        //cout << "TIME: " << elapsed_secs << "s" << endl; \



    MX_ARRAY_OPERATION_AA_CUDA(add, 0)
    MX_ARRAY_OPERATION_AA_CUDA(subtract, 0)
    MX_ARRAY_OPERATION_AA_CUDA(multiply, 0)
    MX_ARRAY_OPERATION_AA_CUDA(divide, 1)


} // namespace NN
} // namespace MX
} // namespace internal

