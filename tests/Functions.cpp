#include <NN/Functions.hpp>
#include <iostream>

int main () {

    std::cout << NN::Activation::ReLU(16) << std::endl;
    std::cout << NN::Activation::Sigmoid(3) << std::endl;

}