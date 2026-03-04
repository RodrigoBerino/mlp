// src/activations/sigmoid.hpp
// Phase 2 — Activations: Sigmoid functor.
//
// σ(x) = 1 / (1 + exp(-x))
// σ'(x) = σ(x) * (1 - σ(x))

#pragma once

#include <cmath>

namespace mlp {

template<typename T>
struct Sigmoid {
    // Forward pass: σ(x) = 1 / (1 + exp(-x))
    [[nodiscard]] T operator()(T x) const {
        return T{1} / (T{1} + std::exp(-x));
    }

    // Derivative: σ'(x) = σ(x) * (1 - σ(x))
    [[nodiscard]] T derivative(T x) const {
        const T s = (*this)(x);
        return s * (T{1} - s);
    }
};

} // namespace mlp
