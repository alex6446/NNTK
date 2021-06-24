#pragma once

#include "NNTK/Core/Base.hpp"

namespace NN
{

  class ActivationFunction
  {
  public:

    virtual ~ActivationFunction() = default;

    virtual NDArray function(const NDArray &x) const = 0;
    virtual NDArray derivative(const NDArray &x) const = 0;
    virtual NDArray derivative(const NDArray &x, const NDArray &y) const = 0;

  }; // class ActivationFunction

  class AFNone : public ActivationFunction
  {
  public:

    NDArray
    function(const NDArray &x) const override
    { return x; }

    NDArray
    derivative(const NDArray &x) const override
    { return NDArray::ones(x.shape()); }

    NDArray
    derivative(const NDArray &x, const NDArray &y) const override
    { return derivative(x); };

  }; // class AFNone

  class AFSigmoid : public ActivationFunction
  {
  public:

    NDArray
    function(const NDArray &x) const override;

    NDArray
    derivative(const NDArray &x) const override
    { return function(x) * (1. - function(x)); }

    NDArray
    derivative(const NDArray &x, const NDArray &y) const override
    { return y * (1. - y); };

  }; // class AFSigmoid

  class AFReLU : public ActivationFunction
  {
  public:

    NDArray
    function(const NDArray &x) const override;

    NDArray
    derivative(const NDArray &x) const override;

    NDArray
    derivative(const NDArray &x, const NDArray &y) const override
    { return derivative(x); }

  }; // class AFReLU

  class AFLeakyReLU : public ActivationFunction
  {
  public:

    AFLeakyReLU(nn_type alpha)
      : m_alpha(alpha)
    { }

    NDArray
    function(const NDArray &x) const override;

    NDArray
    derivative(const NDArray &x) const override;

    NDArray
    derivative(const NDArray &x, const NDArray &y) const override
    { return derivative(x); }

  private:

      nn_type m_alpha;

  }; // class AFLeakyReLU


  namespace Activation
  {

    extern ActivationFunction *None;
    extern ActivationFunction *Sigmoid;
    extern ActivationFunction *ReLU;
    ActivationFunction *LeakyReLU(nn_type alpha);

  } // namespace Activation

} // namespace NN
