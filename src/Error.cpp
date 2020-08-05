#include "NN/Error.hpp"

namespace NN {

    BaseError::BaseError() : std::runtime_error("NN: Unknown Error") {}
    BaseError::BaseError(const std::string& msg) : std::runtime_error("NN:" + msg) {}
    BaseError::BaseError(const std::string& type, const std::string& msg) : std::runtime_error("NN:" + type + msg) {}


    MatrixError::MatrixError(const std::string& msg) : BaseError(":MatrixError:", msg) {}

}