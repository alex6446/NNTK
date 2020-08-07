#include <NN/Matrix.hpp>
#include <NN/Functions.hpp>
#include <iostream>
#include <ctime>


void perceptron () {
    using namespace std;
    using namespace NN;

    MX::Matrixf training_inputs = {{0, 0, 1},
                                   {1, 1, 1},
                                   {1, 0, 1},
                                   {0, 1, 1}};
                                            
    MX::Matrixf training_outputs = MX::Matrixf({0, 1, 1, 0}).transpose();

    MX::Matrixf synaptic_weights = MX::Matrixf(3, 1).randomize(-1, 1);

    cout << "Случайные инициализирующие веса:" << endl;
    cout << synaptic_weights << endl;    

    MX::Matrixf outputs;
    MX::Matrixf input_layer;
    // Метод обратного распространения
    for (int i = 0; i < 20000; i++) { 
        input_layer = training_inputs;
        outputs = MX::Dot(input_layer, synaptic_weights).apply(NN::Activation::Sigmoid, 0);

        MX::Matrixf err = training_outputs - outputs;
        MX::Matrixf adjustments = MX::Dot(input_layer.transpose(), err * (outputs * (1 - outputs)));

        synaptic_weights += adjustments;
    }

    cout << "Веса после обучения:" << endl;
    cout << synaptic_weights << endl;

    cout << "Результат после обучения: " << endl;
    cout << outputs << endl;

    // TEST
    MX::Matrixf new_inputs = {1, 1, 0};
    MX::Matrixf output = MX::Dot(new_inputs, synaptic_weights).apply(NN::Activation::Sigmoid, 0);

    cout << "Новая ситуация: " << endl;
    cout << output << endl;
}

int main () {

    using namespace std;

    clock_t begin = clock();
    srand(time(NULL));

    perceptron();

    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "TIME: " << elapsed_secs << "s" << endl;

}