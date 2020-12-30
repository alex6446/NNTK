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

        class Array : public Base {
        public:
            Array(const std::string& msg);
        };

    }

}
