//#include "NN/Funcs.hpp"
#include "../include/NN/Funcs.hpp"

namespace NN {

    namespace Activation
    {

        MX::Array<float> (*None)(const MX::Array<float> &, int, float) = nullptr;

    } // namespace Activation

    namespace Loss
    {

        const MX::Array<float> (*None)(const MX::Array<float> &, const MX::Array<float> &, int, float) = nullptr;

        const MX::Array<float>
        MSE(const MX::Array<float> &A, const MX::Array<float> &Y, int mode, float hp)
        {
            MX::Array<float> loss;
            switch (mode) {
                case 0:
                    loss = MX::Sum((A - Y) * (A - Y), 1);
                    loss /= A.cols();
                    break;
                case 1:
                    loss =  2 * (A - Y) / A.cols();
                    break;
                default: abort();
            }
            return loss;
        }

        const Array<float>
        BCE (const MX::Array<float> &A, const MX::Array<float> &Y, int mode, float hp) {
            MX::Array<float>* loss = new MX::Array<float>(A.rows(), A.cols());
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

    } // namespace Loss

} // namespace NN
