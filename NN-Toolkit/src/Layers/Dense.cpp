#include "NN/Layers/Dense.hpp"

namespace NN {

    namespace Layer {

        Dense::Dense (
            int neurons,
            float (*activation) (float, int, float),
            bool bias,
            int rand_from,
            int rand_to,
            float hyperparameter
        ) : size(neurons),
            rand_a(rand_from), 
            rand_b(rand_to)
        {
            g = activation;
            this->bias = bias;
            hp = hyperparameter;
            bound = false;
        }

        void Dense::forwardProp (const void* X) {
            this->X = (MX::Matrixf*)X;
            Z = MX::Dot(W, *(this->X));
            if (bias) {
                for (int i = 0; i < Z.rows(); ++i)
                    for (int j = 0; j < Z.cols(); ++j)
                        Z(i, j) += b(i, 0);
            }
            
            if (g) A = Z.apply(g, 0, hp);
            else A = Z;
        }

        void Dense::backProp (const void* gradient) {
            dZ = *((MX::Matrixf*)gradient);
            delete (MX::Matrixf*)gradient;
            if (g) dZ *= Z.apply(g, 1, hp);
            dW = MX::Dot(dZ, X->transpose()) / dZ.cols();
            if (bias) db = MX::Sum(dZ, 1) / dZ.cols();
        }

        void Dense::update (float learning_rate) {
            W -= learning_rate * dW;
            if (bias)
                b -= learning_rate * db;
        }

        void Dense::bind (const std::vector<int>& dimensions) {
            if (bound) return;
            W = MX::Matrixf(size, dimensions[0]).randomize(rand_a, rand_b);
            if (bias)
                b = MX::Matrixf(size, 1).randomize(rand_a, rand_b);
            bound = true;
        }

        void Dense::save (std::string file) const {
            std::ofstream fout(file);
            fout << *this;
            fout.close();
        }

        void Dense::load (std::string file) {
            std::ifstream fin(file);
            fin >> *this;
            fin.close();
        }

        std::ostream& operator<< (std::ostream& os, const Dense& l) {
            os << "Layer Dense {" << std::endl;
            os << "size " << l.size << std::endl;
            std::string g = "None";
            if (l.g == (float (*)(float, int, float))Activation::Sigmoid<float>) g = "Sigmoid";
            if (l.g == (float (*)(float, int, float))Activation::ReLU<float>) g = "ReLU";
            os << "g " << g << std::endl;
            os << "bound " << l.bound << std::endl;
            os << "rand_a " << l.rand_a << std::endl;
            os << "rand_b " << l.rand_b << std::endl;
            os << "hp " << l.hp << std::endl;
            os << "bias " << l.bias << std::endl;
            if (l.bias) os << "b " << l.b << std::endl;
            os << "W " << l.W << std::endl;
            os << "}" << std::endl;
            return os;
        }

        std::istream& operator>> (std::istream& is, Dense& l) {
            std::string buffer;
            is >> buffer; // Layer
            is >> buffer; // Dense
            is >> buffer; // {
            while (buffer != "}") {
                is >> buffer;
                if (buffer == "size") is >> l.size;
                else if (buffer == "g") {
                    is >> buffer;
                    if (buffer == "None") l.g = Activation::None;
                    else if (buffer == "Sigmoid") l.g = Activation::Sigmoid;
                    else if (buffer == "ReLU") l.g = Activation::ReLU;
                }
                else if (buffer == "bound") is >> l.bound;
                else if (buffer == "rand_a") is >> l.rand_a;
                else if (buffer == "rand_b") is >> l.rand_b;
                else if (buffer == "hp") is >> l.hp;
                else if (buffer == "bias") is >> l.bias;
                else if (buffer == "b") is >> l.b;
                else if (buffer == "W") is >> l.W;
            }
            if (l.bias && !l.b.rows() || l.W.rows()) l.bound = false;
            return is;
        }

    } // namespace Layer

} // namespace NN
