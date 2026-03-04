// src/loss/cross_entropy.hpp
// Phase 4 — Loss: Multiclass Cross Entropy for Softmax output.
//
// Loss:
//   L = -Σ_i  y_i * log(max(ŷ_i, ε))
//
// Combined gradient for Softmax + CrossEntropy at z level:
//   δ(L) = ŷ - y
//
// The combined gradient absorbs the Softmax Jacobian, yielding the
// clean element-wise form used directly by Layer::backward_output_z().
//
// Stability: log(0) is avoided via max(ŷ_i, kEpsilon).

#pragma once

#include "core/vector.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace mlp {

template<typename T>
struct CrossEntropy {
    // Smallest value substituted for ŷ_i to prevent log(0)
    static constexpr T kEpsilon = T{1e-12};

    // ------------------------------------------------------------------
    // compute_loss: L = -Σ_i  y_i * log(max(ŷ_i, ε))
    //
    // y     : one-hot ground-truth vector
    // y_hat : predicted probability vector (output of Softmax)
    // ------------------------------------------------------------------
    [[nodiscard]] T compute_loss(const Vector<T>& y_hat,
                                 const Vector<T>& y) const {
        check_sizes(y_hat, y, "compute_loss");

        T loss = T{0};
        for (std::size_t i = 0; i < y.size(); ++i) {
            if (y[i] > T{0}) {
                const T safe_p = y_hat[i] > kEpsilon ? y_hat[i] : kEpsilon;
                loss -= y[i] * std::log(safe_p);
            }
        }
        return loss;
    }

    // ------------------------------------------------------------------
    // compute_delta: δ(L) = ŷ - y
    //
    // This is the combined gradient ∂L/∂z for Softmax + CrossEntropy.
    // Pass the result directly to Layer::backward_output_z().
    // ------------------------------------------------------------------
    [[nodiscard]] Vector<T> compute_delta(const Vector<T>& y_hat,
                                          const Vector<T>& y) const {
        check_sizes(y_hat, y, "compute_delta");
        return y_hat - y;
    }

private:
    static void check_sizes(const Vector<T>& y_hat,
                             const Vector<T>& y,
                             const char* op) {
        if (y_hat.size() != y.size()) {
            throw std::invalid_argument(
                std::string("CrossEntropy::") + op +
                ": size mismatch (" +
                std::to_string(y_hat.size()) + " vs " +
                std::to_string(y.size()) + ")");
        }
        if (y_hat.empty()) {
            throw std::invalid_argument(
                std::string("CrossEntropy::") + op + ": empty vectors");
        }
    }
};

} // namespace mlp
