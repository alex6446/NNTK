#include <ctime>
#include <iostream>

#include "../Funcs.hpp"
#include "Array.hpp"

int
main()
{
    srand(time(NULL));
    std::cout << "Start ing" << std::endl;
    NN::MX::Array<float> a = NN::MX::Random<float>({3, 4}, -4, 6);
    std::cout << a << std::endl;

    std::cout << NN::Activation::Sigmoid(a) << std::endl; 
    std::cout << NN::Activation::Sigmoid(a, NN::Activation::Mode::Derivative) << std::endl; 

    std::cout << NN::Activation::ReLU(a) << std::endl; 
    std::cout << NN::Activation::ReLU(a, NN::Activation::Mode::Derivative) << std::endl; 

    std::cout << NN::Loss::MSE(NN::MX::Random<float>({3, 4}), NN::MX::Random<float>({3, 4})) << std::endl;
    std::cout << NN::Loss::BCE(NN::MX::Random<float>({3, 4}), NN::MX::Random<float>({3, 4})) << std::endl;

    //model.add(Layer::Conv2D::Builder().;
}
