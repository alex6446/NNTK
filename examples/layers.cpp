#include "../Layers/Dense.hpp"
#include "../Layers/Conv2DNew.hpp"
#include <iostream>
#include <iterator>

int
main()
{
    std::cout << ((NN::Layer::Base *)(NN::Layer::Dense::Builder()
                    .activation(NN::Activation::Sigmoid)
                    .neurons(3)
                    .rand_range(0.1, 0.3)
                 ))->bind({16, 8})->shape() << std::endl;
    std::cout << ((NN::Layer::Base *)(NN::Layer::Conv2D::Builder()
                    .activation(NN::Activation::ReLU)
                    .rand_range(0.1, 0.3)
                    .filters(4)
                    .padding(1)
                 ))->bind({8, 3, 400, 400})->shape() << std::endl;
}
