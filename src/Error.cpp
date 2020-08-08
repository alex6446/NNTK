#include "NN/Error.hpp"

namespace NN {

    namespace Error {

        Base::Base() : std::runtime_error("NN::Error: Unknown Error") {}
        Base::Base(const std::string& msg) : std::runtime_error("NN::Error:" + msg) {}
        Base::Base(const std::string& type, const std::string& msg) : std::runtime_error("NN::Error:" + type + msg) {}


        Matrix::Matrix(const std::string& msg) : Base(":Matrix:", msg) {}

    }

}