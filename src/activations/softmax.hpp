// src/activations/softmax.hpp
// Phase 2 — Activations: Softmax functor (specialised for Vector<T>).
//
// Numerically stable formulation:
//   softmax(z)_i = exp(z_i - max(z)) / Σ_j exp(z_j - max(z))
//
// The Jacobian of Softmax is:
//   ∂s_i/∂z_j = s_i * (δ_ij - s_j)
//
// For the combined Softmax + CrossEntropy loss, the upstream gradient
// simplifies to (ŷ - y), handled in the loss layer — NOT here.
// Here we expose jacobian_times_vec() for standalone gradient checking.

#pragma once

#include "core/vector.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mlp {

template<typename T>
struct Softmax {
    // Forward pass: returns probability vector.
    // Uses exp(z_i - max(z)) for numerical stability.
    [[nodiscard]] Vector<T> operator()(const Vector<T>& z) const {
        if (z.empty()) {
            throw std::invalid_argument("Softmax: input vector is empty");
        }

        // Find max for numerical stability
        T max_val = z[0];
        for (std::size_t i = 1; i < z.size(); ++i) {
            if (z[i] > max_val) { max_val = z[i]; }
        }

        // Compute shifted exponentials
        Vector<T> out(z.size());
        T sum = T{0};
        for (std::size_t i = 0; i < z.size(); ++i) {
            out[i] = std::exp(z[i] - max_val);
            sum += out[i];
        }

        // Normalise
        for (std::size_t i = 0; i < out.size(); ++i) {
            out[i] /= sum;
        }

        return out;
    }

    // Jacobian–vector product: J(z) * v
    // J_ij = s_i*(δ_ij - s_j)  →  (Jv)_i = s_i*(v_i - s·v)
    // Useful for gradient checking without materialising the full Jacobian.
    [[nodiscard]] Vector<T> jacobian_times_vec(const Vector<T>& z,
                                               const Vector<T>& v) const {
        if (z.size() != v.size()) {
            throw std::invalid_argument(
                "Softmax::jacobian_times_vec: size mismatch");
        }
        const Vector<T> s = (*this)(z);
        const T sv = s.dot(v);            // scalar: s · v
        Vector<T> result(z.size());
        for (std::size_t i = 0; i < z.size(); ++i) {
            result[i] = s[i] * (v[i] - sv);
        }
        return result;
    }
};

} // namespace mlp
