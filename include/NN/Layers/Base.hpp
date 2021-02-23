#pragma once

#include <vector>
#include <iostream>
#include <fstream>

//#include "NN/MX/Array.hpp"
//#include "NN/Funcs.hpp"
//#include "NN/Core.hpp"

#include "../Core.hpp"
#include "../Funcs.hpp"
#include "../MX/Array.hpp"

#define FUNC_DEF(func) #func

namespace NN
{
namespace Layer
{

    class Base
    {
    protected:

        using size_type = typename MX::Array<nn_type>::size_type;
        using activation_function_type = MX::Array<nn_type> (*)(const MX::Array<nn_type> &, Activation::Mode, float);

    public:

        virtual ~Base() = default;

        virtual Base * forwardprop(const MX::Array<nn_type> &input) = 0;
        virtual Base * backprop(const MX::Array<nn_type> &gradient) = 0;

        virtual Base *
        forwardprop(const Base *layer)
        { forwardprop(layer->output()); return this; }

        virtual
        Base *
        backprop(const Base *layer)
        { backprop(layer->gradient()); return this; }

        virtual Base * update(float learning_rate) = 0;
        virtual Base * bind(const MX::Array<size_type> &shape) = 0;

        virtual
        Base *
        bind(const Base *layer)
        { bind(layer->output_shape()); return this; }

        virtual inline
        Base *
        reset()
        { m_is_bound = false; return this; }

        virtual const MX::Array<nn_type> & output() const = 0;
        virtual MX::Array<nn_type> gradient() const = 0;
        virtual MX::Array<size_type> output_shape() const = 0;

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

            operator Base * () const
            { return m_layer; }

        protected:

            Base *m_layer;

        };

    protected:

        bool m_is_bound;

    };

} // namespace Layer

} // namespace NN
