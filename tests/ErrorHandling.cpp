#include <iostream>

#include "NN/Error.hpp"

void errorChecker () {
    // throw NN::Error::Base();
    // throw NN::Error::Base(":errorChecker: program failed successfully");
    throw NN::Error::Matrix(":errorChecker: program failed successfully");
}

int main () {
    errorChecker();
}

