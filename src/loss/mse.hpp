// src/loss/mse.hpp
// Phase 4 — Loss: Mean Squared Error (secondary loss function).
//
// Loss:
//   L = (1/n) * Σ_i (ŷ_i - y_i)²
//
// Gradient ∂L/∂ŷ:
//   g_i = (2/n) * (ŷ_i - y_i)
//
// The gradient is passed to Layer::backward_output() which then
// multiplies by φ'(z) to obtain δ(L).

#pragma once

#include "core/vector.hpp"

#include <cstddef>
#include <stdexcept>

namespace mlp {

template<typename T>
struct MSE {
    // ------------------------------------------------------------------
    // compute_loss: L = (1/n) * Σ_i (ŷ_i - y_i)²
    // ------------------------------------------------------------------
    [[nodiscard]] T compute_loss(const Vector<T>& y_hat,
                                 const Vector<T>& y) const {
        check_sizes(y_hat, y, "compute_loss");

        T sum = T{0};
        for (std::size_t i = 0; i < y.size(); ++i) {
            const T diff = y_hat[i] - y[i];
            sum += diff * diff;
        }
        return sum / static_cast<T>(y.size());
    }

    // ------------------------------------------------------------------
    // compute_gradient: g_i = (2/n) * (ŷ_i - y_i)
    //
    // This is ∂L/∂ŷ — pass to Layer::backward_output() so it
    // multiplies by φ'(z) to compute δ(L).
    // ------------------------------------------------------------------
    [[nodiscard]] Vector<T> compute_gradient(const Vector<T>& y_hat,
                                              const Vector<T>& y) const {
        check_sizes(y_hat, y, "compute_gradient");

        const T scale = T{2} / static_cast<T>(y.size());
        Vector<T> grad(y.size());
        for (std::size_t i = 0; i < y.size(); ++i) {
            grad[i] = scale * (y_hat[i] - y[i]);
        }
        return grad;
    }

private:
    static void check_sizes(const Vector<T>& y_hat,
                             const Vector<T>& y,
                             const char* op) {
        if (y_hat.size() != y.size()) {
            throw std::invalid_argument(
                std::string("MSE::") + op +
                ": size mismatch (" +
                std::to_string(y_hat.size()) + " vs " +
                std::to_string(y.size()) + ")");
        }
        if (y_hat.empty()) {
            throw std::invalid_argument(
                std::string("MSE::") + op + ": empty vectors");
        }
    }
};

} // namespace mlp
