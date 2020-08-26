#pragma once

#include "NN/Matrix.hpp"
#include "NN/Error.hpp"
#include "NN/Layers/Base.hpp"
#include "NN/Functions.hpp"

namespace NN {

    namespace Layer {

        class Flatten : public Base {
        private:

            // neurons for the whole dataset
            MX::Matrixf Z; 
            MX::Matrixf A;
            MX::Matrixf dZ;

            const std::vector<MX::Image>* X;

            // total number of neurons
            int size;

            // random initialization range
            int rand_a;
            int rand_b;

        public:

            Flatten (
                float (*activation) (float, int, float) = Activation::None,
                bool bias = false,
                int rand_from = -1,
                int rand_to = 1,
                float hyperparameter = 1
            );

            inline Flatten* sActivation (float (*g) (float, int, float)) { this->g = g; return this; }
            inline Flatten* sBias (bool bias) { this->bias = bias; return this; }
            inline Flatten* sRandFrom (bool rand_a) { this->rand_a = rand_a; return this; }
            inline Flatten* sRandTo (bool rand_b) { this->rand_b = rand_b; return this; }
            inline Flatten* sHyperparameter (bool hp) { this->hp = hp; return this; }

            void forwardProp (const void* X) override;
            void backProp (const void* gradient) override;
            void update (float learning_rate) override;
            void bind (const std::vector<int>& dimensions) override;

            inline const void* getA () const override { return &A; }
            const void* getGradient () const override;
            inline std::vector<int> getDimensions () const override { return { size }; }
        };

    } // namespace Layer

} // namespace NN