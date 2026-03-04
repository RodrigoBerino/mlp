// src/activations/relu.hpp
// Phase 2 — Activations: ReLU functor.
//
// ReLU(x) = max(0, x)
// ReLU'(x) = 1 if x > 0, else 0
// Note: derivative at x=0 is defined as 0 (subgradient convention).

#pragma once

namespace mlp {

template<typename T>
struct ReLU {
    // Forward pass: max(0, x)
    [[nodiscard]] T operator()(T x) const {
        return x > T{0} ? x : T{0};
    }

    // Derivative: 1 if x > 0, else 0
    [[nodiscard]] T derivative(T x) const {
        return x > T{0} ? T{1} : T{0};
    }
};

} // namespace mlp
