#pragma once

#include <stdexcept>
#include <string>

namespace NN {

    class BaseError : public std::runtime_error {
    public:
        BaseError();
        BaseError(const std::string& msg);
        BaseError(const std::string& type, const std::string& msg);
    };

    class MatrixError : public BaseError {
    public:
        MatrixError(const std::string& msg);
    };

}
