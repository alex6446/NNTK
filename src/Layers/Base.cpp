#include "NN/Layers/Base.hpp"

namespace NN {

    namespace Layer {

        void Base::forwardProp (const MX::Matrixf& X) {

        }

        void Base::backProp (const MX::Matrixf& gradient) {

        }

        MX::Matrixf const& Base::getA () const {
            return MX::Matrixf();
        }

        MX::Matrixf Base::getGradient () const {
            return MX::Matrixf();
        }


    }

}