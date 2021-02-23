#include <iostream>
#include <ostream>
#include <string>

#include <cstdlib>
#include <cxxabi.h>

#include "ArrayInternal.hpp"

template<typename T>
std::string type_name()
{
    int status;
    std::string tname = typeid(T).name();
    char *demangled_name = abi::__cxa_demangle(tname.c_str(), NULL, NULL, &status);
    if(status == 0) {
        tname = demangled_name;
        std::free(demangled_name);
    }   
    return tname;
}

int
main()
{
    using namespace nn::mx::internal; 
    using value_type = float;
    std::cout << typeid(value_type).name() << std::endl;
    std::cout << typeid(int).name() << std::endl;
    std::cout << typeid(double).name() << std::endl;
    std::cout << typeid(long double).name() << std::endl;
    std::cout << typeid(unsigned int).name() << std::endl;
    std::cout << typeid(std::string).name() << std::endl;
    std::cout << type_name<value_type>() << std::endl;
    std::cout << type_name<int>() << std::endl;
    std::cout << type_name<double>() << std::endl;
    std::cout << type_name<long double>() << std::endl;
    std::cout << type_name<unsigned int>() << std::endl;
    std::cout << type_name<std::string>() << std::endl;
    std::cout << type_to_string<value_type>() << std::endl;
    std::cout << type_to_string<int>() << std::endl;
    std::cout << type_to_string<double>() << std::endl;
    std::cout << type_to_string<long double>() << std::endl;
    std::cout << type_to_string<unsigned int>() << std::endl;
    std::cout << type_to_string<std::string>() << std::endl;


}
