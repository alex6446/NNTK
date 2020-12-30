#include <iostream>
#include <iterator>
#include <ostream>
#include "../Funcs.hpp"
#include "Array.hpp"

int
main()
{
    std::cout << "Start ing" << std::endl;
    NN::MX::Array<float> a = NN::MX::Random<float>({3, 4}, -4, 6);
    std::cout << a << std::endl;

    std::cout << NN::Activation::Sigmoid(a) << std::endl; 
    std::cout << NN::Activation::Sigmoid(a, 1) << std::endl; 

    std::cout << NN::Activation::ReLU(a) << std::endl; 
    std::cout << NN::Activation::ReLU(a, 1) << std::endl; 
}
