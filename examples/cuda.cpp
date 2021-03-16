#include <ostream>
#define NN_GPU_ENABLED

#include "NN/MX/Array.hpp"
#include <iostream>

void
test_speed()
{
    NN::MX::Array<float> a = NN::MX::Array<float>::random({100000000}, 100, 10000);
    NN::MX::Array<float> b = NN::MX::Array<float>::random({100000000}, 100, 10000);
    //std::cout << a << std::endl;
    //std::cout << b << std::endl;
    //std::cout << a + b << std::endl;
    int device = -1;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(a.data(), sizeof(NN::nn_type)*a.size(), device, NULL);
    NN::MX::Array<float> result;
    result = a + b;
    result = a - b;
}

int
main()
{
    test_speed();
}
