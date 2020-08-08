#include <NN/Toolkit.hpp>

#include <iostream>

int main () {
    using namespace NN;
    srand(time(NULL));

    MX::Matrixf X = {{0, 0, 1},
                     {1, 1, 1},
                     {1, 0, 1},
                     {0, 1, 1}};
                     
    MX::Matrixf Y = MX::Matrixf({0, 1, 1, 0}).transpose();
                                

    Sequential model;
    model.add(new Layer::Dense(1, Activation::Sigmoid, true));
    model.fit(X, Y, Loss::MSE, 4, 20000);
    std::cout << model.predict(X.transpose()) << std::endl;

}