#pragma once

#include <vector>

#include "NN/Matrix.hpp"
#include "NN/Error.hpp"
#include "NN/Layers/Base.hpp"
#include "NN/Functions.hpp"

namespace NN {

    namespace Layer {

        class MaxPooling2D : public Base {
        private:

            // volumes for the whole dataset
            std::vector<MX::Image> Z; 
            std::vector<MX::Image> A;
            std::vector<MX::Image> dZ;

            const std::vector<MX::Image>* X;
            std::vector<int> Xdims; // X dimensions

            int f; // pool dimensions
            int p; // padding
            int s; // stride

            // random initialization range
            int rand_a;
            int rand_b;

        public:

            MaxPooling2D (
                int pool_size=2,
                int padding=0,
                int stride=2,
                float (*activation) (float, int, float) = Activation::None,
                bool bias = false,
                int rand_from = -1,
                int rand_to = 1,
                float hyperparameter = 1
            );

            void forwardProp (const void* X) override;
            void backProp (const void* gradient) override;
            void update (float learning_rate) override;
            void bind (const std::vector<int>& dimensions) override;

            inline const void* getA () const override { return &A; }
            const void* getGradient () const override;
            std::vector<int> getDimensions () const override;
        };

    } // namespace Layer

} // namespace NN
