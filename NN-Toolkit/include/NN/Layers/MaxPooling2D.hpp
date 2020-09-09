#pragma once

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
            float rand_a;
            float rand_b;

        public:

            MaxPooling2D (
                int pool_size=2,
                int padding=0,
                int stride=2,
                float (*activation) (float, int, float) = Activation::None,
                bool bias = false,
                float rand_from = -1,
                float rand_to = 1,
                float hyperparameter = 1
            );

            inline MaxPooling2D* sPoolSize (int f) { this->f = f; return this; }
            inline MaxPooling2D* sPadding (int p) { this->p = p; return this; }
            inline MaxPooling2D* sStride (int s) { this->s = s; return this; }
            inline MaxPooling2D* sActivation (float (*g) (float, int, float)) { this->g = g; return this; }
            inline MaxPooling2D* sBias (bool bias) { this->bias = bias; return this; }
            inline MaxPooling2D* sRandFrom (float rand_a) { this->rand_a = rand_a; return this; }
            inline MaxPooling2D* sRandTo (float rand_b) { this->rand_b = rand_b; return this; }
            inline MaxPooling2D* sHyperparameter (float hp) { this->hp = hp; return this; }

            void forwardProp (const void* X) override;
            void backProp (const void* gradient) override;
            void update (float learning_rate) override;
            void bind (const std::vector<int>& dimensions) override;

            inline const void* getA () const override { return &A; }
            const void* getGradient () const override;
            std::vector<int> getDimensions () const override;

            inline void print () const override { std::cout << *this; }
            void save (std::string file) const override;
            void load (std::string file) override;
            inline void output (std::ostream& os) override { os << *this; }
            inline void input (std::istream& is) override { is >> *this; }

            friend std::ostream& operator<< (std::ostream& os, const MaxPooling2D& l);
            friend std::istream& operator>> (std::istream& is, MaxPooling2D& l);

        };

    } // namespace Layer

} // namespace NN
