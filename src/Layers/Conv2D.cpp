#include "NN/Layers/Conv2D.hpp"

namespace NN {

    namespace Layer {

        Conv2D::Conv2D (
            int filters,
            int filter_size,
            int padding,
            int stride,
            float (*activation) (float, int, float),
            bool bias,
            int rand_from,
            int rand_to,
            float hyperparameter
        ) : size(filters),
            f(filter_size),
            p(padding),
            s(stride),
            rand_a(rand_from), 
            rand_b(rand_to)
        {
            g = activation;
            this->bias = bias;
            bound = false;
        }

        void Conv2D::forwardProp (const void* X) {
            this->X = (std::vector<MX::Image>*)X;
            int bs = this->X->size(); // batch size
            int cs = this->X->begin()->size(); // number of channels / filter depth
            int zm = ((*(this->X))[0][0].rows() + 2*p - f) / s + 1;
            int zn = ((*(this->X))[0][0].cols() + 2*p - f) / s + 1;
            Z = std::vector<MX::Image>(bs, MX::Image(size, MX::Matrixf(zm, zn)));
            for (int i = 0; i < bs; ++i) // loop through each sample
                for (int j = 0; j < size; ++j) { // loop through each filter
                    for (int k = 0; k < cs; ++k) // loop through each channel
                        Z[i][j] += MX::Convolve((*(this->X))[i][k], W[j][k], p, s);
                    if (bias) // one bias value for each filter / output channel
                        for (int bi = 0; bi < zm; ++bi)
                            for (int bj = 0; bj < zn; ++bj)
                                Z[i][j](bi, bj) += b(j, 0);
                }
            if (g) {
                A = std::vector<MX::Image>(bs, MX::Image(size));
                for (int i = 0; i < bs; ++i)
                    for (int j = 0; j < size; ++j)
                        A[i][j] = Z[i][j].apply(g, 0, hp);
            } else A = Z;
        }

        void Conv2D::backProp (const void* gradient) {
            dZ = *((std::vector<MX::Image>*)gradient);
            delete (std::vector<MX::Image>*)gradient;
            int bs = this->X->size(); // batch size
            int cs = this->X->begin()->size(); // number of channels / filter depth
            if (g) {
                for (int i = 0; i < dZ.size(); ++i)
                    for (int j = 0; j < dZ[0].size(); ++j)
                        dZ[i][j] *= Z[i][j].apply(g, 1, hp);
            }
            dW = std::vector<MX::Filter>(W.size(), MX::Filter(W[0].size(), MX::Matrixf(f, f)));
            if (bias)
                db = MX::Matrixf(b.rows(), b.cols());
            
            for (int i = 0; i < bs; ++i) // loop through each sample
                for (int j = 0; j < size; ++j) { // loop through each filter
                    for (int k = 0; k < cs; ++k) // loop through each channel
                        for (int xi = -p, zi = 0; zi < dZ[0][0].rows(); xi+=s, ++zi)
                            for (int xj = -p, zj = 0; zj < dZ[0][0].cols(); xj+=s, ++zj) 
                                for (int fi = 0; fi < f; ++fi)
                                    for (int fj = 0; fj < f; ++fj)
                                        dW[j][k](fi, fj) += (*(this->X))[i][k].get(xi+fi, xj+fj) * dZ[i][j](zi, zj);
                    if (bias)
                        db(j, 0) += MX::Sum(dZ[i][j]);
                }
            for (int j = 0; j < size; ++j) // loop through each filter
                for (int k = 0; k < cs; ++k) // loop through each channel
                    dW[j][k] /= bs;
            if (bias)
                db /= bs;
        }

        void Conv2D::update (float learning_rate) {
            for (int i = 0; i < W.size(); ++i)
                for (int j = 0; j < W[0].size(); ++j)
                    W[i][j] -= learning_rate * dW[i][j];
            if (bias)
                b -= learning_rate * db;
        }

        void Conv2D::bind (const std::vector<int>& dimensions) {
            if (bound) return;
            W = std::vector<MX::Filter>(size, MX::Filter(dimensions[0], MX::Matrixf(f, f).randomize(rand_a, rand_b)));
            Xdims = dimensions;
            if (bias)
                b = MX::Matrixf(size, 1).randomize(rand_a, rand_b);
            bound = true;
        }

        const void* Conv2D::getGradient () const {
            int bs = this->X->size(); // batch size
            int cs = this->X->begin()->size(); // number of channels / filter depth
            std::vector<MX::Image>* dX = new std::vector<MX::Image>(bs, MX::Image(cs, 
                MX::Matrixf(X->begin()->begin()->rows(), X->begin()->begin()->cols()))); // ziroed gradient
            for (int i = 0; i < bs; ++i) // loop through each sample
                for (int j = 0; j < size; ++j) // loop through each filter
                    for (int k = 0; k < cs; ++k) // loop through each channel
                        for (int xi = -p, zi = 0; zi < dZ[0][0].rows(); xi+=s, ++zi)
                            for (int xj = -p, zj = 0; zj < dZ[0][0].cols(); xj+=s, ++zj) 
                                for (int fi = 0; fi < f; ++fi)
                                    for (int fj = 0; fj < f; ++fj)
                                        (*(dX))[i][k].add(xi+fi, xj+fj, W[j][k](fi, fj) * dZ[i][j](zi, zj));
            return dX;
        }

        std::vector<int> Conv2D::getDimensions () const {
            std::vector<int> Adims = {
                size, // number of channels
                (Xdims[1] + 2*p - f) / s + 1, // A m
                (Xdims[2] + 2*p - f) / s + 1 // A n
            };
            return Adims; 
        }

    } // namespace Layer

} // namespace NN
