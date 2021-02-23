#include <iostream>
#include <iterator>
#include <ostream>

#include "../Layers/Conv2DNew.hpp"
#include "Array.hpp"
#include "ArrayBase.hpp"

//int main () {
    //using namespace NN;
    //srand(time(NULL));

    //MX::Array<float> X = {{0, 1, 1, 0},
                          //{0, 1, 0, 1},
                          //{1, 1, 1, 1}};
                    
    //MX::Array<float> Y = MX::Array<float>({0, 1, 1, 0}).reshape({1, 4});
                                
    //Layer::Base *layer = new Layer::Dense(1, Activation::Sigmoid);
    //layer->bind({X.shape()});
    //for (auto i : MX::Sequence<int>({500})) {
        //layer->forwardprop(X);
        //layer->backprop(Loss::MSE(layer->output(), Y, Loss::Mode::Derivative));
        //layer->update(0.5f);
        //std::cout << layer->output() << std::endl;
        //std::cout << Loss::MSE(layer->output(), Y) << std::endl;
    //}

//}
int main () {
    using namespace NN;
    //srand(time(NULL));
    MX::Array<nn_type> X = { {
        { 
            { 3, 5, 3, 2, 2 }, 
            { 2, 5, 2, 1, 3 }, 
            { 5, 1, 3, 4, 1 }, 
            { 4, 2, 4, 1, 4 } 
        }, { 
            { 3, 1, 4, 5, 3 }, 
            { 1, 2, 4, 4, 5 }, 
            { 4, 4, 5, 3, 1 }, 
            { 3, 4, 2, 1, 3 } 
        } 
    } };
    MX::Array<nn_type> Y = { {
        { 
            { 5, 2, 2 }, 
            { 5, 1, 2 } 
        }, { 
            { 4, 2, 1 }, 
            { 2, 2, 4 } 
        }
    } };
    // W = {
    //     {
    //         { 
    //             { 3, 3, 4 }, 
    //             { 5, 1, 3 }, 
    //             { 2, 3, 5 } 
    //         }, { 
    //             { 2, 2, 2 }, 
    //             { 5, 4, 4 }, 
    //             { 1, 5, 3 } 
    //         }
    //     }, {
    //         { 
    //             { 1, 1, 2 }, 
    //             { 2, 1, 3 }, 
    //             { 5, 1, 4 } 
    //         }, { 
    //             { 2, 1, 1 }, 
    //             { 1, 5, 2 }, 
    //             { 4, 4, 2 } 
    //         }
    //     }
    // };
    Layer::Conv2D lc(2, 3, 1, 2, Activation::Sigmoid, true, -2, 2);
    lc.bind(X.shape());
    lc.forwardprop(X);
    std::cout << "A: " << lc.output() << std::endl;
    lc.backprop(Y);
    std::cout << "dX: " << lc.gradient() << std::endl;
    lc.update(2.f);
    std::cout << lc << std::endl;

}
