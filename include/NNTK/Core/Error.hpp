#pragma once

#include <stdexcept>

#define NN_RUNTIME_ERROR(condition, message) { if (condition) { throw std::runtime_error(std::string(__FUNCTION__) + ":" + message); } }

