#include "NN/Error.hpp"

namespace NN {

    BaseError::BaseError() : std::runtime_error("Unknown Error") { }
    BaseError::BaseError(const std::string& str) : std::runtime_error(str) { }
    BaseError::BaseError(const std::string& str1, const std::string& str2) : std::runtime_error(str1 + str2) { }

}