#include "NN/Layers/Dense.hpp"

namespace NN {

    namespace Layer {

        Dense::Dense (
            int neurons,
            float (*activation) (float, int, float),
            bool bias,
            int rand_from,
            int rand_to,
            float hyperparameter
        ) : size(neurons),
            rand_a(rand_from), 
            rand_b(rand_to)
        {
            g = activation;
            this->bias = bias;
            bound = false;
        }

        void Dense::forwardProp (const void* X) {
            this->X = (MX::Matrixf*)X;
            Z = MX::Dot(W, *(this->X));
            if (bias) {
                for (int i = 0; i < Z.rows(); ++i)
                    for (int j = 0; j < Z.cols(); ++j)
                        Z(i, j) += b(i, 0);
            }
            
            if (g) A = Z.apply(g, 0, hp);
            else A = Z;
        }

        void Dense::backProp (const void* gradient) {
            dZ = *((MX::Matrixf*)gradient);
            delete (MX::Matrixf*)gradient;
            if (g) dZ *= Z.apply(g, 1, hp);
            dW = MX::Dot(dZ, X->transpose()) / dZ.cols();
            if (bias) db = MX::Sum(dZ, 1) / dZ.cols();
        }

        void Dense::update (float learning_rate) {
            W -= learning_rate * dW;
            if (bias)
                b -= learning_rate * db;
        }

        void Dense::bind (const std::vector<int>& dimensions) {
            if (bound) return;
            W = MX::Matrixf(size, dimensions[0]).randomize(rand_a, rand_b);
            if (bias)
                b = MX::Matrixf(size, 1).randomize(rand_a, rand_b);
            bound = true;
        }

    } // namespace Layer

} // namespace NN
