#include "NN/Layers/Dense.hpp"

#include <iostream>
#include <ostream>

namespace NN
{
namespace Layer
{

Dense::
Dense(size_type neurons_count,
      ActivationFunction *activation,
      bool is_bias_enabled,
      float rand_from,
      float rand_to,
      float hyperparam)
: m_neurons_count(neurons_count)
, m_rand_from(rand_from)
, m_rand_to(rand_to)
, m_is_bias_enabled(is_bias_enabled)
, m_activation(activation)
{ m_is_bound = false; }

Base *
Dense::
forwardprop(const NDArray &input)
{
    m_input = &input;
    m_net = NDArray::dot(input, m_weights);
    if (m_is_bias_enabled)
        m_net = NDArray::sum(m_net, m_bias);
    m_output = m_activation->function(m_net);
    return this;
}

Base *
Dense::
backprop(const NDArray &gradient)
{
    m_dnet = gradient * m_activation->derivative(m_net, m_output);
    m_dweights = NDArray::dot(m_input->t(), m_dnet) / m_dnet.shape(0);
    if (m_is_bias_enabled)
        m_dbias = NDArray::sum(m_dnet, 0, false) / m_dnet.shape(0);
    return this;
}

Base *
Dense::
update(float learning_rate)
{
    m_weights -= learning_rate * m_dweights;
    if (m_is_bias_enabled)
        m_bias -= learning_rate * m_dbias;
    return this;
}

Base *
Dense::
bind(const NDSize &shape)
{
    if (m_is_bound)
        return this;
    m_weights = NDArray::random({shape(1), m_neurons_count}, m_rand_from, m_rand_to);
    m_input_shape = shape;
    if (m_is_bias_enabled)
        m_bias = NDArray::random({m_neurons_count}, m_rand_from, m_rand_to);
    m_is_bound = true;
    //m_weights = {{ -0.56177, -0.576535, 0.40228 }};
    //m_bias = {{ -0.807144 }};
    return this;
}

const Base *
Dense::
save(std::string file) const
{
    std::ofstream fout(file);
    fout << *this;
    fout.close();
    return this;
}

Base *
Dense::
load(std::string file)
{
    std::ifstream fin(file);
    fin >> *this;
    fin.close();
    return this;
}

std::ostream &
operator<<(std::ostream &os, const Dense &layer) {
    os << "Layer Dense {" << std::endl
       << "neurons_count: " << layer.m_neurons_count << std::endl;
    std::string activation = "none";
    if (layer.m_activation == Activation::Sigmoid) activation = "Sigmoid";
    if (layer.m_activation == Activation::ReLU) activation = "ReLU";
    os << "activation: " << activation << std::endl
       << "is_bound: " << layer.m_is_bound << std::endl
       << "rand_from: " << layer.m_rand_from << std::endl
       << "rand_to: " << layer.m_rand_to << std::endl
       << "is_bias_enabled: " << layer.m_is_bias_enabled << std::endl
       << "input_shape: " << layer.m_input_shape << std::endl
       << "bias: " << layer.m_bias << std::endl
       << "weights: " << layer.m_weights << std::endl
       << "}" << std::endl;
    return os;
}

std::istream &
operator>>(std::istream &is, Dense &layer) {
    std::string buffer;
    is >> buffer // Layer
       >> buffer // Dense
       >> buffer; // {
    while (buffer != "}") {
        is >> buffer;
        if (buffer == "neurons_count:") is >> layer.m_neurons_count;
        else if (buffer == "activation:") {
            is >> buffer;
            if (buffer == "none") layer.m_activation = Activation::None;
            else if (buffer == "sigmoid") layer.m_activation = Activation::Sigmoid;
            else if (buffer == "relu") layer.m_activation = Activation::ReLU;
        }
        else if (buffer == "is_bound:") is >> layer.m_is_bound;
        else if (buffer == "rand_from:") is >> layer.m_rand_from;
        else if (buffer == "rand_to:") is >> layer.m_rand_to;
        else if (buffer == "m_is_bias_enabled:") is >> layer.m_is_bias_enabled;
        //else if (buffer == "input_shape:") is >> layer.m_input_shape;
        //else if (buffer == "bias:") is >> layer.m_bias;
        //else if (buffer == "weights:") is >> layer.m_weights;
    }
    if (layer.m_input_shape.size() == 0)
        layer.m_is_bound = false;
    return is;
}

} // namespace Layer

} // namespace NN
