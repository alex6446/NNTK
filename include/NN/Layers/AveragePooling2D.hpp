#pragma once

#include <vector>

#include "NN/Matrix.hpp"
#include "NN/Error.hpp"
#include "NN/Layers/Base.hpp"
#include "NN/Functions.hpp"

namespace NN {

    namespace Layer {

        class AveragePooling2D : public Base {
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

            AveragePooling2D (
                int pool_size=2,
                int padding=0,
                int stride=2,
                float (*activation) (float, int, float) = Activation::None,
                bool bias = false,
                int rand_from = -1,
                int rand_to = 1,
                float hyperparameter = 1
            );

            inline AveragePooling2D* sPoolSize (int f) { this->f = f; return this; }
            inline AveragePooling2D* sPadding (bool p) { this->p = p; return this; }
            inline AveragePooling2D* sStride (bool s) { this->s = s; return this; }
            inline AveragePooling2D* sActivation (float (*g) (float, int, float)) { this->g = g; return this; }
            inline AveragePooling2D* sBias (bool bias) { this->bias = bias; return this; }
            inline AveragePooling2D* sRandFrom (bool rand_a) { this->rand_a = rand_a; return this; }
            inline AveragePooling2D* sRandTo (bool rand_b) { this->rand_b = rand_b; return this; }
            inline AveragePooling2D* sHyperparameter (bool hp) { this->hp = hp; return this; }

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
