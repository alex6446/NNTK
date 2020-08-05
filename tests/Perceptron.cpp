#include <NN/Matrix.hpp>
#include <NN/Functions.hpp>
#include <iostream>
#include <ctime>


void perceptron () {
    using namespace std;

    NN::Matrixf training_inputs = {{0, 0, 1},
                                   {1, 1, 1},
                                   {1, 0, 1},
                                   {0, 1, 1}};
                                            
    NN::Matrixf training_outputs = NN::Matrixf({0, 1, 1, 0}).transpose();

    NN::Matrixf synaptic_weights = NN::Matrixf(3, 1).randomize(-1, 1);

    cout << "Случайные инициализирующие веса:" << endl;
    cout << synaptic_weights << endl;    

    NN::Matrixf outputs;
    NN::Matrixf input_layer;
    // Метод обратного распространения
    for (int i = 0; i < 20000; i++) { 
        input_layer = training_inputs;
        outputs = input_layer.dot(synaptic_weights).apply(NN::Activation::Sigmoid, 0);

        NN::Matrixf err = training_outputs - outputs;
        NN::Matrixf adjustments = input_layer.transpose().dot(err * (outputs * (1 - outputs)));

        synaptic_weights += adjustments;
    }

    cout << "Веса после обучения:" << endl;
    cout << synaptic_weights << endl;

    cout << "Результат после обучения: " << endl;
    cout << outputs << endl;

    // TEST
    NN::Matrixf new_inputs = {1, 1, 0};
    NN::Matrixf output = new_inputs.dot(synaptic_weights).apply(NN::Activation::Sigmoid, 0);

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