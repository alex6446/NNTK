#include <NN/Error.hpp>
#include <iostream>

void errorChecker () {
    //throw NN::BaseError();
    // throw NN::BaseError("errorChecker() error");
    throw NN::BaseError("Test::ErrorHandling: ", "errorChecker() error");
}

int main () {
    errorChecker();
}

