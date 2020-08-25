#include "NN/Layers/Flatten.hpp"

namespace NN {

    namespace Layer {

        Flatten::Flatten (
            float (*activation) (float, int, float),
            bool bias,
            int rand_from,
            int rand_to,
            float hyperparameter
        ) : rand_a(rand_from), 
            rand_b(rand_to)
        {
            g = activation;
            this->bias = bias;
        }

        void Flatten::forwardProp (const void* X) {
            this->X = (std::vector<MX::Image>*)X;
            Z = MX::Matrixf(size, this->X->size());
            for (int i = 0; i < (*(this->X)).size(); ++i)
                for (int j = 0, zj = 0; j < (*(this->X))[0].size(); ++j)
                    for (int xi = 0; xi < (*(this->X))[0][0].rows(); ++xi)
                        for (int xj = 0; xj < (*(this->X))[0][0].cols(); ++xj, ++zj)
                            Z(zj, i) = (*(this->X))[i][j](xi, xj);            
                            
            if (bias) {
                for (int i = 0; i < Z.rows(); ++i)
                    for (int j = 0; j < Z.cols(); ++j)
                        Z(i, j) += b(i, 0);
            }
            
            if (g) A = Z.apply(g, 0, hp);
            else A = Z;
        }

        void Flatten::backProp (const void* gradient) {
            dZ = *((MX::Matrixf*)gradient);
            delete (MX::Matrixf*)gradient;
            if (g) dZ *= Z.apply(g, 1, hp);
            if (bias) db = MX::Sum(dZ, 1) / dZ.cols();
        }

        void Flatten::update (float learning_rate) {
            if (bias)
                b -= learning_rate * db;
        }

        void Flatten::bind (const std::vector<int>& dimensions) {
            size = dimensions[0] * dimensions[1] * dimensions[2];
            if (bias)
                b = MX::Matrixf(size, 1).randomize(rand_a, rand_b);
        }

        const void* Flatten::getGradient () const {
            std::vector<MX::Image>* dX = new std::vector<MX::Image>((*X).size(), 
                MX::Image((*X)[0].size(), MX::Matrixf((*X)[0][0].rows(), (*X)[0][0].cols())));
            for (int i = 0; i < (*X).size(); ++i)
                for (int j = 0, zj = 0; j < (*X)[0].size(); ++j)
                    for (int xi = 0; xi < (*X)[0][0].rows(); ++xi)
                        for (int xj = 0; xj < (*X)[0][0].cols(); ++xj, ++zj)
                            (*dX)[i][j](xi, xj) = dZ(zj, i);
            return dX;
        }


    } // namespace Layer

} // namespace NN
