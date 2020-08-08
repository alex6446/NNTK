#include <NN/Functions.hpp>

namespace NN {

    namespace Loss {

        template <>
        MX::Matrixf MSE<MX::Matrixf> (const MX::Matrixf& A, const MX::Matrixf& Y, int mode, float hp) {
            switch (mode) {
                case 0: {
                    MX::Matrixf loss(1, A.cols());
                    loss = MX::Sum((A - Y) * (A - Y), 0);
                    return loss /= A.rows();
                }
                case 1: return 2 * (A - Y) / A.rows();
                default: abort();
            }
        }

    }

}