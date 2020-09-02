#include <NN/Toolkit.hpp>

#include <iostream>

void CNN () {
    using namespace NN;
    srand(time(NULL));

    std::vector<MX::Image> X(8);
                     
    MX::Matrixf Y = MX::Matrixf({
        { 1, 0, 0, 1, 0, 1, 0, 1 },
        { 0, 1, 1, 0, 1, 0, 1, 0 }
    });

    for (int i = 0; i < X.size(); ++i) {
        X[i].push_back(MX::Matrixf(28, 28).randomize(0, 1));             
        X[i].push_back(MX::Matrixf(28, 28).randomize(0, 1));             
        X[i].push_back(MX::Matrixf(28, 28).randomize(0, 1));             
    }            

    Sequential model;
    model.add((new Layer::Conv2D())->sFilters(4)->sPadding(1));
    model.add((new Layer::MaxPooling2D()));
    model.add((new Layer::Conv2D())->sFilters(10)->sPadding(2));
    model.add((new Layer::AveragePooling2D()));
    model.add((new Layer::Flatten())->sActivation(Activation::ReLU)->sBias(true));
    model.add((new Layer::Dense())->sNeurons(10));
    model.add((new Layer::Dense())->sNeurons(2));
    model.fit(X, Y, Loss::MSE, 2, 200);
    model.fit(X, Y, Loss::MSE, 2, 2);
    std::cout << model.predict(X) << std::endl;
    model.save("tests/files/CNN.model");
    model.reset();
    model.fit(X, Y, Loss::MSE, 2, 2);
    std::cout << model.predict(X) << std::endl;

    Sequential model2;
    model2.load("tests/files/CNN.model");
    std::cout << model2.predict(X) << std::endl;
}

int main () {
    using namespace std;

    clock_t begin = clock();
    srand(time(NULL));

    CNN();

    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "TIME: " << elapsed_secs << "s" << endl;
}