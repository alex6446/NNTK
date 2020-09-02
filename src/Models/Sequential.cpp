#include "NN/Models/Sequential.hpp"

namespace NN {

    Sequential::Sequential () {}
    Sequential::~Sequential () {
        for (int i = 0; i < L.size(); ++i)
            delete L[i];
    }

    void Sequential::add (Layer::Base* layer) {
        L.push_back(layer);
    }

    void Sequential::fit (
        const MX::Matrixf& X, 
        const MX::Matrixf& Y,
        const void* (*l) (const MX::Matrixf&, const MX::Matrixf&, int, float),
        int batch_size,
        int epochs,
        float learning_rate,
        float hyperparameter
    ) {
        build({ X.rows() });
        int epoch = 0;
        MX::Matrixf bX(X.rows(), batch_size); // batch input
        MX::Matrixf bY(Y.rows(), batch_size); // batch output
        int sample = 0; // current sample in database
        
        while (epoch < epochs) {
            for (int j = 0; j < bX.cols(); ++j) {
                for (int i = 0; i < bX.rows(); ++i)
                    bX(i, j) = X(i, sample);
                for (int i = 0; i < bY.rows(); ++i)
                    bY(i, j) = Y(i, sample);
                ++sample;
                if (sample >= X.cols()) {
                    ++epoch;
                    sample = 0;
                }
            }

            // forward propagation
            L[0]->forwardProp(&bX);
            for (int i = 1; i < L.size(); ++i)
                L[i]->forwardProp(L[i-1]);
            
            // back propagation
            L[L.size()-1]->backProp(l(*((const MX::Matrixf*)L[L.size()-1]->getA()), bY, 1, hyperparameter));
            for (int i = L.size()-2; i >= 0; --i)
                L[i]->backProp(L[i+1]);

            // update weights
            for (int i = 0; i < L.size(); ++i)
                L[i]->update(learning_rate);
        }
    }

    void Sequential::fit (
        const std::vector<MX::Image>& X, 
        const MX::Matrixf& Y,
        const void* (*l) (const MX::Matrixf&, const MX::Matrixf&, int, float),
        int batch_size,
        int epochs,
        float learning_rate,
        float hyperparameter
    ) {
        build({ (int)X[0].size(), X[0][0].rows(), X[0][0].cols() });
        int epoch = 0;
        std::vector<MX::Image> bX(batch_size); // batch input
        MX::Matrixf bY(Y.rows(), batch_size); // batch output
        int sample = 0; // current sample in database
        
        while (epoch < epochs) {
            bX.clear();
            for (int j = 0; j < batch_size; ++j) {
                bX.push_back(X[sample]);
                for (int i = 0; i < bY.rows(); ++i)
                    bY(i, j) = Y(i, sample);
                ++sample;
                if (sample >= X.size()) {
                    ++epoch;
                    sample = 0;
                }
            }

            // forward propagation
            L[0]->forwardProp(&bX);
            for (int i = 1; i < L.size(); ++i)
                L[i]->forwardProp(L[i-1]);
            
            // back propagation
            L[L.size()-1]->backProp(l(*((const MX::Matrixf*)L[L.size()-1]->getA()), bY, 1, hyperparameter));
            for (int i = L.size()-2; i >= 0; --i)
                L[i]->backProp(L[i+1]);

            // update weights
            for (int i = 0; i < L.size(); ++i)
                L[i]->update(learning_rate);
        }
    }

    void Sequential::build (const std::vector<int>& dimensions) {
        L[0]->bind(dimensions);
        for (int i = 1; i < L.size(); ++i)
            L[i]->bind(L[i-1]);
    }

    MX::Matrixf Sequential::predict (const MX::Matrixf& X) {
        // forward propagation
        L[0]->forwardProp(&X);
        for (int i = 1; i < L.size(); ++i)
            L[i]->forwardProp(L[i-1]->getA());
        return *((MX::Matrixf*)(L[L.size()-1]->getA()));
    }

    MX::Matrixf Sequential::predict (const std::vector<MX::Image>& X) {
        // forward propagation
        L[0]->forwardProp(&X);
        for (int i = 1; i < L.size(); ++i)
            L[i]->forwardProp(L[i-1]->getA());
        return *((MX::Matrixf*)(L[L.size()-1]->getA()));
    }

    std::vector<MX::Image> Sequential::predict2D (const std::vector<MX::Image>& X) {
        // forward propagation
        L[0]->forwardProp(&X);
        for (int i = 1; i < L.size(); ++i)
            L[i]->forwardProp(L[i-1]->getA());
        return *((std::vector<MX::Image>*)(L[L.size()-1]->getA()));
    }

    void Sequential::save (std::string file) const {
        std::ofstream fout(file);
        fout << *this;
        fout.close();
    }

    void Sequential::load (std::string file) {
        std::ifstream fin(file);
        fin >> *this;
        fin.close();
    }

    std::ostream& operator<< (std::ostream& os, const Sequential& m) {
        os << "Model Sequential {" << std::endl;
        for (auto l : m.L)
            l->output(os);
        os << "}" << std::endl;
        return os;            
    }

    std::istream& operator>> (std::istream& is, Sequential& m) {
        m.L.clear();
        std::string buffer;
        is >> buffer; // Model
        is >> buffer; // Sequential
        is >> buffer; // {
        int pos;
        while (buffer != "}") {
            pos = is.tellg();
            is >> buffer;
            if (buffer == "Layer") {
                is >> buffer;
                is.seekg(pos);
                if (buffer == "Dense") m.L.push_back(new Layer::Dense);
                else if (buffer == "Conv2D") m.L.push_back(new Layer::Conv2D);
                else if (buffer == "AveragePooling2D") m.L.push_back(new Layer::AveragePooling2D);
                else if (buffer == "MaxPooling2D") m.L.push_back(new Layer::MaxPooling2D);
                else if (buffer == "Flatten") m.L.push_back(new Layer::Flatten);
                m.L.back()->input(is);
            }
        }
        return is;
    }


}