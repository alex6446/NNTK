#pragma once

#include <stdexcept>
#include <string>

namespace NN {

    class BaseError : public std::runtime_error {
    public:
        BaseError();
        BaseError(const std::string& str);
        BaseError(const std::string& str1, const std::string& str2);
    };

}
