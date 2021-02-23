#define NN_GPU_ENABLED

#include "NN/MX/Array.hpp"
#include <iostream>

void
test_speed()
{
    NN::MX::Array<float> a = NN::MX::Random<float>({100000000}, 100, 10000);
    NN::MX::Array<float> b = NN::MX::Random<float>({100000000}, 100, 10000);
    //std::cout << a << std::endl;
    //std::cout << b << std::endl;
    //std::cout << a + b << std::endl;
    NN::MX::Array<float> result;
    result = a + b;
    result = a + b;

}

int
main()
{
    test_speed();
}
