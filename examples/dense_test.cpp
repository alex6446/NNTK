#include <iostream>
#include <iterator>
#include <ostream>

#include "../Layers/Dense.hpp"
#include "Array.hpp"

int main () {
    using namespace NN;
    srand(time(NULL));

    MX::Array<float> X = {{0, 0, 1},
                          {1, 1, 1},
                          {1, 0, 1},
                          {0, 1, 1}};

    MX::Array<float> Y = MX::Array<float>({0, 1, 1, 0}).reshape({4, 1});

    Layer::Base *layer = new Layer::Dense(1, Activation::Sigmoid);
    layer->bind({X.shape()});
    for (auto i : MX::Sequence<int>({2000})) {
        layer->forwardprop(X);
        layer->backprop(Loss::MSE(layer->output(), Y, Loss::Mode::Derivative));
        layer->update(0.5f);
        std::cout << layer->output() << std::endl;
        std::cout << Loss::MSE(layer->output(), Y) << std::endl;
    }
    std::cout << layer->gradient() << std::endl;

}
