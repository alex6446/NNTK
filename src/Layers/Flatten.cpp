#include "NN/Layers/Flatten.hpp"

namespace NN {

    namespace Layer {

        Flatten::Flatten (
            float (*activation) (float, int, float),
            bool bias,
            float rand_from,
            float rand_to,
            float hyperparameter
        ) : rand_a(rand_from), 
            rand_b(rand_to)
        {
            g = activation;
            this->bias = bias;
            hp = hyperparameter;
            bound = false;
        }

        void Flatten::forwardProp (const void* X) {
            this->X = (std::vector<MX::Image>*)X;
            Z = MX::Matrixf(size, this->X->size());
            for (int i = 0; i < (*(this->X)).size(); ++i)
                for (int j = 0, zj = 0; j < (*(this->X))[0].size(); ++j)
                    for (int xi = 0; xi < (*(this->X))[0][0].rows(); ++xi)
                        for (int xj = 0; xj < (*(this->X))[0][0].cols(); ++xj, ++zj)
                            Z(zj, i) = (*(this->X))[i][j](xi, xj);            
                            
            if (bias) {
                for (int i = 0; i < Z.rows(); ++i)
                    for (int j = 0; j < Z.cols(); ++j)
                        Z(i, j) += b(i, 0);
            }
            
            if (g) A = Z.apply(g, 0, hp);
            else A = Z;
        }

        void Flatten::backProp (const void* gradient) {
            dZ = *((MX::Matrixf*)gradient);
            delete (MX::Matrixf*)gradient;
            if (g) dZ *= Z.apply(g, 1, hp);
            if (bias) db = MX::Sum(dZ, 1) / dZ.cols();
        }

        void Flatten::update (float learning_rate) {
            if (bias)
                b -= learning_rate * db;
        }

        void Flatten::bind (const std::vector<int>& dimensions) {
            if (bound) return;
            size = dimensions[0] * dimensions[1] * dimensions[2];
            if (bias)
                b = MX::Matrixf(size, 1).randomize(rand_a, rand_b);
            bound = true;
        }

        const void* Flatten::getGradient () const {
            std::vector<MX::Image>* dX = new std::vector<MX::Image>((*X).size(), 
                MX::Image((*X)[0].size(), MX::Matrixf((*X)[0][0].rows(), (*X)[0][0].cols())));
            for (int i = 0; i < (*X).size(); ++i)
                for (int j = 0, zj = 0; j < (*X)[0].size(); ++j)
                    for (int xi = 0; xi < (*X)[0][0].rows(); ++xi)
                        for (int xj = 0; xj < (*X)[0][0].cols(); ++xj, ++zj)
                            (*dX)[i][j](xi, xj) = dZ(zj, i);
            return dX;
        }

        void Flatten::save (std::string file) const {
            std::ofstream fout(file);
            fout << *this;
            fout.close();
        }

        void Flatten::load (std::string file) {
            std::ifstream fin(file);
            fin >> *this;
            fin.close();
        }

        std::ostream& operator<< (std::ostream& os, const Flatten& l) {
            os << "Layer Flatten {" << std::endl;
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
            os << "}" << std::endl;
            return os;
        }

        std::istream& operator>> (std::istream& is, Flatten& l) {
            std::string buffer;
            is >> buffer; // Layer
            is >> buffer; // Flatten
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
            }
            if (l.bias && !l.b.rows()) l.bound = false;
            return is;
        }

    } // namespace Layer

} // namespace NN
