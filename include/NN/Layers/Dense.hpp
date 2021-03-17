#pragma once

#include "NN/Layers/Base.hpp"
#include "NN/Functions/Activation.hpp"

namespace NN
{
namespace Layer
{

class Dense : public Base
{
public:

    Dense(size_type neurons_count=1,
          ActivationFunction *activation=Activation::Sigmoid,
          bool is_bias_enabled=true,
          float rand_from=-1,
          float rand_to=1,
          float hyperparam=1);

    Base * forwardprop(const NDArray &input) override;
    Base * backprop(const NDArray &gradient) override;
    Base * update(float learning_rate) override;
    Base * bind(const NDSize &shape) override;

    const NDArray & output() const override { return m_output; }
    NDArray gradient() const override { return NDArray(NDArray::dot(m_dnet, m_weights.t())); }
    NDSize output_shape() const override { return { m_input_shape(1), m_neurons_count }; }

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
        activation(ActivationFunction *activation)
        {
            ((Dense *)m_layer)->m_activation = activation;
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

    NDArray m_weights;
    NDArray m_dweights;

    NDArray m_net;
    NDArray m_dnet;
    NDArray m_output;

    const NDArray *m_input;
    NDSize m_input_shape;

    size_type m_neurons_count;

    float m_rand_from;
    float m_rand_to;

    bool m_is_bias_enabled;
    NDArray m_bias;
    NDArray m_dbias;

    ActivationFunction *m_activation;

};

} // namespace Layer

} // namespace NN
