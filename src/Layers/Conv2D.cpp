#include "NN/Layers/Conv2D.hpp"

#include <iterator>
#include <ostream>

namespace NN
{
namespace Layer
{

Conv2D::
Conv2D(int filters,
       int filter_size,
       int padding,
       int stride,
       activation_function_type activation,
       bool is_bias_enabled,
       float rand_from,
       float rand_to,
       float hyperparam)
: m_filters_count(filters)
, m_filter_size(filter_size)
, m_padding(padding)
, m_stride(stride)
, m_rand_from(rand_from)
, m_rand_to(rand_to)
, m_is_bias_enabled(is_bias_enabled)
, m_activation(activation)
, m_hyperparam(hyperparam)
{ m_is_bound = false; }

Base *
Conv2D::
forwardprop(const MX::Array<nn_type> &input)
{
    m_input = &input;
    m_net = MX::Fill<nn_type>(this->output_shape(), 0);
    for (int i = 0; i < m_net.shape(0); ++i) // loop through each sample
        for (int j = 0; j < m_net.shape(1); ++j) { // loop through each filter
            for (int k = 0; k < input.shape(1); ++k) // loop through each channel
                m_net[i][j] += MX::Convolve(input[i][k], m_weights[j][k], m_padding, m_stride);
            if (m_is_bias_enabled) // one bias value for each filter / output channel
                for (int bi = 0; bi < m_net.shape(2); ++bi)
                    for (int bj = 0; bj < m_net.shape(3); ++bj)
                        m_net(i, j, bi, bj) += m_bias(j);
        }
    m_output = m_activation(m_net, Activation::Mode::Base, m_hyperparam);
    return this;
}

Base *
Conv2D::
backprop(const MX::Array<nn_type> &gradient)
{
    m_dnet = gradient * m_activation(m_net, Activation::Mode::Derivative, m_hyperparam);
    m_dweights = MX::Fill<nn_type>(m_weights.shape(), 0);
    if (m_is_bias_enabled)
        m_dbias = MX::Fill<nn_type>(m_bias.shape(), 0);

    for (int i = 0; i < m_dnet.shape(0); ++i) { // loop through each sample
        for (int j = 0; j < m_dnet.shape(1); ++j) { // loop through each filter
            for (int k = 0; k < m_input_shape(1); ++k) // loop through each channel
                for (int xi = -m_padding, zi = 0; zi < m_dnet.shape(2); xi+=m_stride, ++zi)
                    for (int xj = -m_padding, zj = 0; zj < m_dnet.shape(3); xj+=m_stride, ++zj)
                        for (int fi = 0; fi < m_filter_size; ++fi)
                            for (int fj = 0; fj < m_filter_size; ++fj)
                                m_dweights(j, k, fi, fj) += m_input->force_get(i, k, xi+fi, xj+fj) * m_dnet(i, j, zi, zj);
            if (m_is_bias_enabled)
                m_dbias(j) += MX::Sum(m_dnet[i][j]);
        }
    }
    m_dweights /= m_dnet.shape(0);
    if (m_is_bias_enabled)
        m_dbias /= m_dnet.shape(0);
    return this;
}

Base *
Conv2D::
update(float learning_rate)
{
    m_weights -= learning_rate * m_dweights;
    if (m_is_bias_enabled)
        m_bias -= learning_rate * m_dbias;
    return this;
}

Base *
Conv2D::
bind(const MX::Array<size_type> &shape)
{
    if (m_is_bound)
        return this;
    m_weights = MX::Random<nn_type>({m_filters_count, shape(1), m_filter_size, m_filter_size}, m_rand_from, m_rand_to);
    m_input_shape = shape;
    if (m_is_bias_enabled)
        m_bias = MX::Random<nn_type>({m_filters_count}, m_rand_from, m_rand_to);
    m_is_bound = true;
    return this;
}

MX::Array<nn_type>
Conv2D::
gradient() const
{
    MX::Array<nn_type> dinput = MX::Fill<nn_type>(m_input->shape(), 0); // ziroed gradient
    for (int i = 0; i < m_dnet.shape(0); ++i) // loop through each sample
        for (int j = 0; j < m_dnet.shape(1); ++j) // loop through each filter
            for (int k = 0; k < m_input_shape(1); ++k) // loop through each channel
                for (int xi = -m_padding, zi = 0; zi < m_dnet.shape(2); xi+=m_stride, ++zi)
                    for (int xj = -m_padding, zj = 0; zj < m_dnet.shape(3); xj+=m_stride, ++zj)
                        for (int fi = 0; fi < m_filter_size; ++fi)
                            for (int fj = 0; fj < m_filter_size; ++fj)
                                dinput.force_add(m_weights(j, k, fi, fj) * m_dnet(i, j, zi, zj), i, k, xi+fi, xj+fj);
    return dinput;
}

const Base *
Conv2D::
save(std::string file) const
{
    std::ofstream fout(file);
    fout << *this;
    fout.close();
    return this;
}

Base *
Conv2D::
load(std::string file)
{
    std::ifstream fin(file);
    fin >> *this;
    fin.close();
    return this;
}

std::ostream &
operator<<(std::ostream &os, const Conv2D &layer)
{
    os << "Layer Conv2D {" << std::endl
       << "filters_count: " << layer.m_filters_count << std::endl
       << "filter_size: " << layer.m_filter_size << std::endl
       << "padding: " << layer.m_padding << std::endl
       << "stride: " << layer.m_stride << std::endl;
    std::string activation = "None";
    if (layer.m_activation == (Conv2D::activation_function_type)Activation::Sigmoid<nn_type>) activation = "Sigmoid";
    if (layer.m_activation == (Conv2D::activation_function_type)Activation::ReLU<nn_type>) activation = "ReLU";
    os << "activation: " << activation << std::endl
       << "is_bound: " << layer.m_is_bound << std::endl
       << "rand_from: " << layer.m_rand_from << std::endl
       << "rand_to: " << layer.m_rand_to << std::endl
       << "hyperparam: " << layer.m_hyperparam << std::endl
       << "is_bias_enabled: " << layer.m_is_bias_enabled << std::endl
       << "input_shape: " << layer.m_input_shape << std::endl;
    if (layer.m_is_bias_enabled)
        os << "bias: " << layer.m_bias << std::endl;
    os << "weights: " << layer.m_weights << std::endl
       << "}" << std::endl;
    return os;
}

std::istream &
operator>>(std::istream& is, Conv2D& layer)
{
    std::string buffer;
    is >> buffer // Layer
       >> buffer // Conv2D
       >> buffer; // {
    while (buffer != "}") {
        is >> buffer;
        if (buffer == "m_filters_count:") is >> layer.m_filters_count;
        else if (buffer == "filter_size:") is >> layer.m_filter_size;
        else if (buffer == "padding:") is >> layer.m_padding;
        else if (buffer == "stride:") is >> layer.m_stride;
        else if (buffer == "activation:") {
            is >> buffer;
            if (buffer == "None") layer.m_activation = Activation::None;
            else if (buffer == "Sigmoid") layer.m_activation = Activation::Sigmoid;
            else if (buffer == "ReLU") layer.m_activation = Activation::ReLU;
        }
        else if (buffer == "is_bound:") is >> layer.m_is_bound;
        else if (buffer == "rand_from:") is >> layer.m_rand_from;
        else if (buffer == "rand_to:") is >> layer.m_rand_to;
        else if (buffer == "hyperparam:") is >> layer.m_hyperparam;
        else if (buffer == "is_bias_enabled:") is >> layer.m_is_bias_enabled;
        //else if (buffer == "input_shape:") is >> layer.m_input_shape;
        //else if (buffer == "bias:") is >> layer.m_bias;
        //else if (buffer == "weights:") is >> layer.m_weights;
    }
    if (layer.m_is_bias_enabled && !layer.m_bias.shape(0) || layer.m_weights.shape(0))
        layer.m_is_bound = false;
    return is;
}

} // namespace Layer

} // namespace NN
