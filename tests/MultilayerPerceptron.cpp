#include <NN/Toolkit.hpp>

#include <iostream>

int main () {
    using namespace NN;
    srand(time(NULL));

    MX::Matrixf X = MX::Matrixf({
        {0, 0, 1},
        {1, 1, 1},
        {1, 0, 1},
        {0, 1, 1}}).transpose();
                     
    MX::Matrixf Y = {0, 1, 1, 0};
                                

    Sequential model;
    model.add(new Layer::Dense(1, Activation::Sigmoid, true));
    model.fit(X, Y, Loss::MSE, 4, 20000);
    model.fit(X, Y, Loss::MSE, 2, 200);
    std::cout << model.predict(X) << std::endl;
    model.reset();
    model.fit(X, Y, Loss::MSE, 4, 20);
    std::cout << model.predict(X) << std::endl;

}