#pragma once

#include <stdexcept>
#include <string>

namespace NN {

    namespace Error {

        class Base : public std::runtime_error {
        public:
            Base();
            Base(const std::string& msg);
            Base(const std::string& type, const std::string& msg);
        };

        class Matrix : public Base {
        public:
            Matrix(const std::string& msg);
        };

    }

}
