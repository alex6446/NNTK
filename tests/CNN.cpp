#include <NN/Toolkit.hpp>

#include <iostream>

int main () {
    using namespace NN;
    srand(time(NULL));

    std::vector<MX::Image> X(8);
                     
    MX::Matrixf Y = MX::Matrixf({
        {1, 0,},
        {0, 1,},
        {0, 1,},
        {1, 0,},
        {0, 1,},
        {1, 0,},
        {0, 1,},
        {1, 0,}
    });
    for (int i = 0; i < X.size(); ++i) {
        X[i].push_back(MX::Matrixf(28, 28).randomize(0, 1));             
        X[i].push_back(MX::Matrixf(28, 28).randomize(0, 1));             
        X[i].push_back(MX::Matrixf(28, 28).randomize(0, 1));             
    }            

    Sequential model;
    model.add(new Layer::Conv2D(4, 3, 1, 1));
    model.add(new Layer::Conv2D(10, 2));
    model.add(new Layer::Flatten());
    model.add(new Layer::Dense(10));
    model.add(new Layer::Dense(2));
    model.fit(X, Y, Loss::MSE, 2, 200);
    std::cout << model.predict(X) << std::endl;

}