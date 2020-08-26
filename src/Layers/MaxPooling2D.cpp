#include "NN/Layers/MaxPooling2D.hpp"

namespace NN {

    namespace Layer {

        MaxPooling2D::MaxPooling2D (
            int pool_size,
            int padding,
            int stride,
            float (*activation) (float, int, float),
            bool bias,
            int rand_from,
            int rand_to,
            float hyperparameter
        ) : f(pool_size),
            p(padding),
            s(stride),
            rand_a(rand_from), 
            rand_b(rand_to)
        {
            g = activation;
            this->bias = bias;
            bound = false;
        }

        void MaxPooling2D::forwardProp (const void* X) {
            this->X = (std::vector<MX::Image>*)X;
            int bs = this->X->size(); // batch size
            int cs = this->X->begin()->size(); // number of channels
            int zm = ((*(this->X))[0][0].rows() + 2*p - f) / s + 1;
            int zn = ((*(this->X))[0][0].cols() + 2*p - f) / s + 1;
            Z = std::vector<MX::Image>(bs, MX::Image(cs, MX::Matrixf(zm, zn)));
            for (int i = 0; i < bs; ++i) // loop through each sample
                for (int j = 0; j < cs; ++j) { // loop through each channel
                    for (int xi = -p, zi = 0; zi < zm; xi+=s, ++zi)
                        for (int xj = -p, zj = 0, mi = 0, mj = 0; zj < zn; xj+=s, ++zj) {
                            mi = xi;
                            mj = xj;
                            for (int fi = 0; fi < f; ++fi)
                                for (int fj = 0; fj < f; ++fj)
                                    if ((*(this->X))[i][j].get(xi+fi, xj+fj) > (*(this->X))[i][j].get(mi, mj)) {
                                        mi = xi+fi;
                                        mj = xj+fj;
                                    }
                            Z[i][j](zi, zj) = (*(this->X))[i][j].get(mi, mj);
                        }
                    if (bias) // one bias value for each filter / output channel
                        for (int bi = 0; bi < zm; ++bi)
                            for (int bj = 0; bj < zn; ++bj)
                                Z[i][j](bi, bj) += b(j, 0);
                }
            if (g) {
                A = std::vector<MX::Image>(bs, MX::Image(cs));
                for (int i = 0; i < bs; ++i)
                    for (int j = 0; j < cs; ++j)
                        A[i][j] = Z[i][j].apply(g, 0, hp);
            } else A = Z;
        }

        void MaxPooling2D::backProp (const void* gradient) {
            dZ = *((std::vector<MX::Image>*)gradient);
            delete (std::vector<MX::Image>*)gradient;
            int bs = this->X->size(); // batch size
            int cs = this->X->begin()->size(); // number of channels / filter depth
            if (g) {
                for (int i = 0; i < dZ.size(); ++i)
                    for (int j = 0; j < dZ[0].size(); ++j)
                        dZ[i][j] *= Z[i][j].apply(g, 1, hp);
            }
            if (bias)
                db = MX::Matrixf(b.rows(), b.cols());
            
            for (int i = 0; i < bs; ++i) // loop through each sample
                for (int j = 0; j < cs; ++j) // loop through each channel
                    if (bias)
                        db(j, 0) += MX::Sum(dZ[i][j]);
            if (bias)
                db /= bs;
        }

        void MaxPooling2D::update (float learning_rate) {
            if (bias)
                b -= learning_rate * db;
        }

        void MaxPooling2D::bind (const std::vector<int>& dimensions) {
            if (bound) return;
            Xdims = dimensions;
            if (bias)
                b = MX::Matrixf(dimensions[0], 1).randomize(rand_a, rand_b);
            bound = true;
        }

        const void* MaxPooling2D::getGradient () const {
            int bs = this->X->size(); // batch size
            int cs = this->X->begin()->size(); // number of channels / filter depth
            std::vector<MX::Image>* dX = new std::vector<MX::Image>(bs, MX::Image(cs, 
                MX::Matrixf(X->begin()->begin()->rows(), X->begin()->begin()->cols()))); // ziroed gradient
            for (int i = 0; i < bs; ++i) // loop through each sample
                for (int j = 0; j < cs; ++j) // loop through each channel
                    for (int xi = -p, zi = 0; zi < dZ[0][0].rows(); xi+=s, ++zi)
                        for (int xj = -p, zj = 0, mi = 0, mj = 0; zj < dZ[0][0].cols(); xj+=s, ++zj) {
                            mi = xi;
                            mj = xj;
                            for (int fi = 0; fi < f; ++fi)
                                for (int fj = 0; fj < f; ++fj)
                                    if ((*(this->X))[i][j].get(xi+fi, xj+fj) > (*(this->X))[i][j].get(mi, mj)) {
                                        mi = xi+fi;
                                        mj = xj+fj;
                                    }
                            (*dX)[i][j].add(mi, mj, dZ[i][j].get(zi, zj));
                        }
            return dX;
        }

        std::vector<int> MaxPooling2D::getDimensions () const {
            std::vector<int> Adims = {
                Xdims[0], // number of channels
                (Xdims[1] + 2*p - f) / s + 1, // A m
                (Xdims[2] + 2*p - f) / s + 1 // A n
            };
            return Adims; 
        }

    } // namespace Layer

} // namespace NN
