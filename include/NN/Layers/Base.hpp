#pragma once

#include "NN/Core/Types.hpp"

namespace NN
{
namespace Layer
{

class Base
{
protected:

    using size_type = typename NDSize::size_type;

public:

    virtual ~Base() = default;

    virtual Base * forwardprop(const Base *layer) { forwardprop(layer->output()); return this; }
    virtual Base * backprop(const Base *layer) { backprop(layer->gradient()); return this; }
    virtual Base * bind(const Base *layer) { bind(layer->output_shape()); return this; }
    virtual Base * reset() { m_is_bound = false; return this; }

    virtual Base * forwardprop(const NDArray &input) = 0;
    virtual Base * backprop(const NDArray &gradient) = 0;
    virtual Base * update(float learning_rate) = 0;
    virtual Base * bind(const NDSize &shape) = 0;

    virtual const NDArray & output() const = 0;
    virtual NDArray gradient() const = 0;
    virtual NDSize output_shape() const = 0;

    virtual const Base * save(std::string file) const = 0;
    virtual Base * load(std::string file) = 0;

public:

    // Since all Builder classes will be stored as Base::Builder,
    // I'm expecting to get slicing of methods.
    // However it shouldn't affect the result
    class Builder
    {
    public:

        Builder() = default;
        virtual ~Builder() = default;

        operator Base * () const { return m_layer; }

    protected:

        Base *m_layer;

    };

protected:

    bool m_is_bound;

};

} // namespace Layer

} // namespace NN
