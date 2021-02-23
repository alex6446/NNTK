#pragma once

#include "NN/Layers/Base.hpp"

namespace NN
{
namespace Layer
{

    class Conv2D : public Base
    {
    public:

        Conv2D(int filters=1,
               int filter_size=3,
               int padding=0,
               int stride=1,
               activation_function_type activation=Activation::ReLU,
               bool bias=true,
               float rand_from=-1,
               float rand_to=1,
               float hyperparam=1);

        Base * forwardprop(const MX::Array<nn_type> &input) override;
        Base * backprop(const MX::Array<nn_type> &gradient) override;
        Base * update(float learning_rate) override;
        Base * bind(const MX::Array<size_type> &shape) override;

        const MX::Array<nn_type> &
        output() const override
        { return m_output; }

        MX::Array<nn_type> gradient() const override;

        MX::Array<size_type>
        output_shape() const override
        {
            return {
                m_input_shape(0),
                m_filters_count, // number of channels
                (m_input_shape(2) + 2 * m_padding - m_filter_size) / m_stride + 1, // output m
                (m_input_shape(3) + 2 * m_padding - m_filter_size) / m_stride + 1 // output n
            };
        }

        const Base * save(std::string file) const override;
        Base * load(std::string file) override;

        friend std::ostream &operator<<(std::ostream &os, const Conv2D &layer);
        friend std::istream &operator>>(std::istream &is, Conv2D& layer);

    public:

        class Builder : public Base::Builder
        {
        public:

            Builder()
            { m_layer = new Conv2D(); }

            Builder &
            bias(bool bias)
            {
                ((Conv2D *)m_layer)->m_is_bias_enabled = bias;
                return *this;
            }

            Builder &
            activation(activation_function_type activation)
            {
                ((Conv2D *)m_layer)->m_activation = activation;
                return *this;
            }

            Builder &
            hyperparam(float hyperparam)
            {
                ((Conv2D *)m_layer)->m_hyperparam = hyperparam;
                return *this;
            }

            Builder &
            rand_range(float from, float to)
            {
                ((Conv2D *)m_layer)->m_rand_from = from;
                ((Conv2D *)m_layer)->m_rand_to = to;
                return *this;
            }

            Builder &
            filters(int filters_count)
            {
                ((Conv2D *)m_layer)->m_filters_count = filters_count;
                return *this;
            }

            Builder &
            filter_size(int filter_size)
            {
                ((Conv2D *)m_layer)->m_filter_size = filter_size;
                return *this;
            }

            Builder &
            padding(int padding)
            {
                ((Conv2D *)m_layer)->m_padding = padding;
                return *this;
            }

            Builder &
            stride(int stride)
            {
                ((Conv2D *)m_layer)->m_stride = stride;
                return *this;
            }

        };

    private:

        MX::Array<nn_type> m_weights;
        MX::Array<nn_type> m_dweights;

        // volumes for the whole dataset
        MX::Array<nn_type> m_net;
        MX::Array<nn_type> m_dnet;
        MX::Array<nn_type> m_output;

        const MX::Array<nn_type> *m_input;
        MX::Array<size_type> m_input_shape;

        int m_filters_count; // number of filters
        int m_filter_size; // filter dimensions
        int m_padding; // padding
        int m_stride; // stride

        // random initialization range
        float m_rand_from;
        float m_rand_to;

        bool m_is_bias_enabled;
        MX::Array<nn_type> m_bias; // bias vector
        MX::Array<nn_type> m_dbias; // bias derivative

        activation_function_type m_activation; // activation function
        float m_hyperparam; // hyperparameter

    };

} // namespace Layer

} // namespace NN
