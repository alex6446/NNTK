#pragma once

#include "NNTK/Layers/Base.hpp"

namespace NN
{
namespace Layer
{

class Flatten : public Base
{
public:

    Flatten ();

    Base * forwardprop(const MX::Array<nn_type> &input) override;

    Base *
    backprop(const MX::Array<nn_type> &gradient) override
    { m_dnet = gradient; return this; }

    Base *
    update(float learning_rate) override
    { return this; }

    Base * bind(const MX::Array<size_type> &shape) override;

    const MX::Array<nn_type> &
    output() const override
    { return m_output; }

    MX::Array<nn_type> gradient() const override
    { return MX::Array<nn_type>(m_dnet).reshape(m_input_shape); }

    MX::Array<size_type>
    output_shape() const override
    { return { m_input_shape(0), m_input_shape.size() / m_input_shape(0) }; }

    const Base * save(std::string file) const override;
    Base * load(std::string file) override;

    friend std::ostream &operator<<(std::ostream &os, const Flatten &layer);
    friend std::istream &operator>>(std::istream &is, Flatten &layer);

private:

    MX::Array<nn_type> m_dnet;
    MX::Array<nn_type> m_output;

    MX::Array<size_type> m_input_shape;

};

} // namespace Layer

} // namespace NN
