#include <NN/Functions.hpp>

namespace NN {

    namespace Activation {

        float (*None) (float, int, float) = nullptr;

    }

    namespace Loss {

        const void* (*None) (const MX::Matrixf&, const MX::Matrixf&, int, float) = nullptr;

        const void* MSE (const MX::Matrixf& A, const MX::Matrixf& Y, int mode, float hp) {
            MX::Matrixf* loss = new MX::Matrixf;
            switch (mode) {
                case 0:
                    *loss = MX::Sum((A - Y) * (A - Y), 1);
                    *loss /= A.cols();
                    break;
                case 1: 
                    *loss =  2 * (A - Y) / A.cols();
                    break;
                default: abort();
            }
            return loss;
        }

        const void* BCE (const MX::Matrixf& A, const MX::Matrixf& Y, int mode, float hp) {
            MX::Matrixf* loss = new MX::Matrixf(A.rows(), A.cols());
            switch (mode) {
                case 0:
                    for (int i = 0; i < A.rows(); ++i)
                        for (int j = 0; j < A.cols(); ++j)
                            (*loss)(i, j) = Y(i, j) * std::log(A(i, j) + (1 - Y(i, j)) * std::log(1 - A(i, j)));
                    *loss = -MX::Sum(*loss, 1) / A.cols();
                    break;
                case 1: 
                    for (int i = 0; i < A.rows(); ++i)
                        for (int j = 0; j < A.cols(); ++j)
                            (*loss)(i, j) = (A(i, j) - Y(i, j)) / (A(i, j) * (1 - A(i, j)));
                    *loss /= A.cols();
                    break;
                default: abort();
            }
            return loss;
        }

    }

}