#pragma once

#include "NN/Matrix.hpp"
#include "NN/Error.hpp"
#include "NN/Layers/Base.hpp"
#include "NN/Functions.hpp"

namespace NN {

    namespace Layer {

        class Dense : public Base {
        private:
            
            MX::Matrixf W;
            MX::Matrixf dW;

            // neurons for the whole dataset
            MX::Matrixf Z; 
            MX::Matrixf A;
            MX::Matrixf dZ;

            const MX::Matrixf* X;

            // number of neurons
            int size;

            // random initialization range
            int rand_a;
            int rand_b;

        public:

            Dense (
                int neurons,
                float (*activation) (float, int, float) = Activation::Sigmoid,
                bool bias = true,
                int rand_from = -1,
                int rand_to = 1,
                float hyperparameter = 1
            );

            void forwardProp (const void* X) override;
            void backProp (const void* gradient) override;
            void update (float learning_rate) override;
            void bind (const std::vector<int>& dimensions) override;

            inline const void* getA () const override { return &A; }
            inline const void* getGradient () const override { return new MX::Matrixf(MX::Dot(W.transpose(), dZ)); }
            inline std::vector<int> getDimensions () const override { return { size }; }
        };

    } // namespace Layer

} // namespace NN
