// src/loss/l2_regularization.hpp
// Phase 11 — L2 Regularization: penalty computation and weight decay.
//
// Objective with L2:
//   L_total = L_data + λ * Σ_l Σ_ij W_l(i,j)²
//
// Gradient at layer l:
//   ∂L_total/∂W_l = ∂L_data/∂W_l + λ * W_l
//
// Weight decay equivalence (for SGD):
//   W ← W - η * (∂L_data/∂W + λ*W)
//       = W * (1 - η*λ) - η * ∂L_data/∂W
//
// apply_weight_decay() applies the decay step after mlp.step() so the
// gradient update is already absorbed.  Biases are deliberately NOT
// decayed (standard practice: regularise weights only, not biases).

#pragma once

#include "mlp/mlp.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace mlp {

// -----------------------------------------------------------------------
// compute_l2_penalty
//
// Returns  λ * Σ_l Σ_ij W_l(i,j)²
// (Frobenius-norm-squared summed over all weight matrices, scaled by λ).
//
// Returns 0 immediately when lambda == 0 to avoid unnecessary iteration.
// -----------------------------------------------------------------------
template<typename T, typename Activation>
[[nodiscard]] T compute_l2_penalty(const MLP<T, Activation>& mlp, T lambda)
{
    if (lambda == T{0}) { return T{0}; }

    T sum_sq = T{0};
    for (std::size_t l = 0; l < mlp.num_layers(); ++l) {
        const auto& W = mlp.layer(l).W();
        for (std::size_t i = 0; i < W.rows(); ++i) {
            for (std::size_t j = 0; j < W.cols(); ++j) {
                sum_sq += W(i, j) * W(i, j);
            }
        }
    }
    return lambda * sum_sq;
}

// -----------------------------------------------------------------------
// apply_weight_decay
//
// Applies  W_l ← W_l * (1 - eta_lambda)  to every weight matrix.
//
// This is the weight-decay form of L2 regularisation:
//   eta_lambda = learning_rate * lambda
//
// Biases are NOT decayed (regularising biases provides negligible benefit
// and can hurt convergence).
//
// Call AFTER mlp.step() so the data-gradient update is already applied.
// -----------------------------------------------------------------------
template<typename T, typename Activation>
void apply_weight_decay(MLP<T, Activation>& mlp, T eta_lambda)
{
    if (eta_lambda == T{0}) { return; }

    if (eta_lambda >= T{1}) {
        throw std::invalid_argument(
            "apply_weight_decay: eta_lambda must be < 1 to avoid weight sign "
            "inversion (got " + std::to_string(static_cast<double>(eta_lambda)) +
            "). Reduce learning_rate or lambda so their product per batch < 1.");
    }

    const T factor = T{1} - eta_lambda;
    for (std::size_t l = 0; l < mlp.num_layers(); ++l) {
        auto& W = mlp.layer(l).W();
        for (std::size_t i = 0; i < W.rows(); ++i) {
            for (std::size_t j = 0; j < W.cols(); ++j) {
                W(i, j) *= factor;
            }
        }
    }
}

// -----------------------------------------------------------------------
// compute_weight_norm_sq
//
// Returns Σ_l Σ_ij W_l(i,j)²  (Frobenius norm squared, not scaled).
// Convenience helper for tests that want to inspect weight magnitudes.
// -----------------------------------------------------------------------
template<typename T, typename Activation>
[[nodiscard]] T compute_weight_norm_sq(const MLP<T, Activation>& mlp)
{
    T sum_sq = T{0};
    for (std::size_t l = 0; l < mlp.num_layers(); ++l) {
        const auto& W = mlp.layer(l).W();
        for (std::size_t i = 0; i < W.rows(); ++i) {
            for (std::size_t j = 0; j < W.cols(); ++j) {
                sum_sq += W(i, j) * W(i, j);
            }
        }
    }
    return sum_sq;
}

} // namespace mlp
