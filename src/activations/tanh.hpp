// src/activations/tanh.hpp
// Phase 2 — Activations: Tanh functor.
//
// tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
// tanh'(x) = 1 - tanh²(x)

#pragma once

#include <cmath>

namespace mlp {

template<typename T>
struct Tanh {
    // Forward pass: tanh(x) using std::tanh for numerical stability
    [[nodiscard]] T operator()(T x) const {
        return std::tanh(x);
    }

    // Derivative: tanh'(x) = 1 - tanh²(x)
    [[nodiscard]] T derivative(T x) const {
        const T t = (*this)(x);
        return T{1} - t * t;
    }
};

} // namespace mlp
