#pragma once

//#include "NN/Layers/Base.hpp"
#include "Base.hpp"

namespace NN
{
namespace Layer
{

    class Dense : public Base
    {
    public:

        Dense(size_type neurons_count=1,
              activation_function_type activation=Activation::Sigmoid,
              bool is_bias_enabled=true,
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

        MX::Array<nn_type>
        gradient() const override
        { return MX::Array<nn_type>(MX::Dot(m_dnet, MX::Transpose(m_weights))); }

        MX::Array<size_type>
        output_shape() const override
        { return { m_input_shape(1), m_neurons_count }; }

        const Base * save(std::string file) const override;
        Base * load(std::string file) override;

        friend std::ostream &operator<<(std::ostream &os, const Dense &layer);
        friend std::istream &operator>>(std::istream &is, Dense &layer);

    public:

        class Builder : public Base::Builder
        {
        public:

            Builder()
            { m_layer = new Dense(); }

            Builder &
            bias(bool bias)
            {
                ((Dense *)m_layer)->m_is_bias_enabled = bias;
                return *this;
            }

            Builder &
            activation(activation_function_type activation)
            {
                ((Dense *)m_layer)->m_activation = activation;
                return *this;
            }

            Builder &
            hyperparam(float hyperparam)
            {
                ((Dense *)m_layer)->m_hyperparam = hyperparam;
                return *this;
            }

            Builder &
            neurons(size_type neurons)
            {
                ((Dense *)m_layer)->m_neurons_count = neurons;
                return *this;
            }

            Builder &
            rand_range(float from, float to)
            {
                ((Dense *)m_layer)->m_rand_from = from;
                ((Dense *)m_layer)->m_rand_to = to;
                return *this;
            }

        };

    private:

        MX::Array<nn_type> m_weights;
        MX::Array<nn_type> m_dweights;

        MX::Array<nn_type> m_net;
        MX::Array<nn_type> m_dnet;
        MX::Array<nn_type> m_output;

        const MX::Array<nn_type> *m_input;
        MX::Array<size_type> m_input_shape;

        size_type m_neurons_count;

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
