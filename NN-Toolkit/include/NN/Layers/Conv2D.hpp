#pragma once

#include "NN/Matrix.hpp"
#include "NN/Error.hpp"
#include "NN/Layers/Base.hpp"
#include "NN/Functions.hpp"

namespace NN {

    namespace Layer {

        class Conv2D : public Base {
        private:
            
            std::vector<MX::Filter> W;
            std::vector<MX::Filter> dW;

            // volumes for the whole dataset
            std::vector<MX::Image> Z; 
            std::vector<MX::Image> A;
            std::vector<MX::Image> dZ;

            const std::vector<MX::Image>* X;
            std::vector<int> Xdims; // X dimensions

            int size; // number of filters 
            int f; // filter dimensions
            int p; // padding
            int s; // stride

            // random initialization range
            float rand_a;
            float rand_b;

        public:

            Conv2D (
                int filters=1,
                int filter_size=3,
                int padding=0,
                int stride=1,
                float (*activation) (float, int, float) = Activation::ReLU,
                bool bias = true,
                float rand_from = -1,
                float rand_to = 1,
                float hyperparameter = 1
            );

            inline Conv2D* sFilters (int size) { this->size = size; return this; }
            inline Conv2D* sFilterSize (int f) { this->f = f; return this; }
            inline Conv2D* sPadding (int p) { this->p = p; return this; }
            inline Conv2D* sStride (int s) { this->s = s; return this; }
            inline Conv2D* sActivation (float (*g) (float, int, float)) { this->g = g; return this; }
            inline Conv2D* sBias (bool bias) { this->bias = bias; return this; }
            inline Conv2D* sRandFrom (float rand_a) { this->rand_a = rand_a; return this; }
            inline Conv2D* sRandTo (float rand_b) { this->rand_b = rand_b; return this; }
            inline Conv2D* sHyperparameter (float hp) { this->hp = hp; return this; }

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

            friend std::ostream& operator<< (std::ostream& os, const Conv2D& l);
            friend std::istream& operator>> (std::istream& is, Conv2D& l);

        };

    } // namespace Layer

} // namespace NN
