#include "NN/Models/Sequential.hpp"

namespace NN {

    Sequential::Sequential () {}
    Sequential::~Sequential () {
        for (int i = 0; i < L.size(); ++i)
            delete L[i];
    }

    void Sequential::add (Layer::Base* layer) {
        L.push_back(layer);
    }

    void Sequential::fit (
        const MX::Matrixf& X, 
        const MX::Matrixf& Y,
        MX::Matrixf (*l) (const MX::Matrixf&, const MX::Matrixf&, int, float),
        int batch_size,
        int epochs,
        float learning_rate,
        float hyperparameter
    ) {
        build({ X.cols() });
        int epoch = 0;
        MX::Matrixf bX(X.cols(), batch_size); // batch input
        MX::Matrixf bY(Y.cols(), batch_size); // batch output
        int sample = 0; // current sample in database
        
        while (epoch < epochs) {
            for (int j = 0; j < bX.cols(); ++j) {
                for (int i = 0; i < bX.rows(); ++i)
                    bX(i, j) = X(sample, i);
                for (int i = 0; i < bY.rows(); ++i)
                    bY(i, j) = Y(sample, i);
                ++sample;
                if (sample >= X.rows()) {
                    ++epoch;
                    sample = 0;
                }
            }
            // forward propagation
            L[0]->forwardProp(bX);
            for (int i = 1; i < L.size(); ++i)
                L[i]->forwardProp(L[i-1]->getA());
            
            // back propagation
            L[L.size()-1]->backProp(l(L[L.size()-1]->getA(), bY, 1, hyperparameter));
            for (int i = L.size()-2; i >= 0; --i)
                L[i]->backProp(L[i+1]->getGradient());

            // update weights
            for (int i = 0; i < L.size(); ++i)
                L[i]->update(learning_rate);
        }
    }

    void Sequential::build (const std::vector<int>& dimensions) {
        L[0]->bind(dimensions);
        for (int i = 1; i < L.size(); ++i)
            L[i]->bind(L[i-1]->getDimensions());
    }

    MX::Matrixf Sequential::predict (const MX::Matrixf& X) {
        // forward propagation
        L[0]->forwardProp(X);
        for (int i = 1; i < L.size(); ++i)
            L[i]->forwardProp(L[i-1]->getA());
        return L[L.size()-1]->getA();
    }

}