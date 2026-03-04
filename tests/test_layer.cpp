// tests/test_layer.cpp
// Phase 3 — Unit tests for mlp::Layer<T, Activation>.

#include "layers/layer.hpp"
#include "activations/sigmoid.hpp"
#include "activations/relu.hpp"
#include "activations/tanh.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"

#include <gtest/gtest.h>
#include <cmath>

using mlp::Layer;
using mlp::Matrix;
using mlp::Vector;
using mlp::Sigmoid;
using mlp::ReLU;
using mlp::Tanh;

static constexpr double kEps    = 1e-9;
static constexpr double kGradEps = 1e-4;  // tolerance for numerical gradient check

// =========================================================================
// Forward Pass
// =========================================================================

TEST(LayerForward, SingleNeuronKnownWeights) {
    // 1 neuron, 2 inputs
    // W = [[0.5, -0.5]],  b = [0.1]
    // input = [1.0, 2.0]
    // z = 0.5*1 + (-0.5)*2 + 0.1 = 0.5 - 1.0 + 0.1 = -0.4
    // a = sigmoid(-0.4)
    Matrix<double> W(1, 2, {0.5, -0.5});
    Vector<double> b{0.1};
    Layer<double, Sigmoid<double>> layer(std::move(W), std::move(b));

    Vector<double> input{1.0, 2.0};
    const auto& a = layer.forward(input);

    const double expected_z = -0.4;
    const double expected_a = 1.0 / (1.0 + std::exp(-expected_z));

    EXPECT_NEAR(layer.z()[0], expected_z, kEps);
    EXPECT_NEAR(a[0], expected_a, kEps);
}

TEST(LayerForward, TwoNeuronsKnownWeights) {
    // 2 neurons, 3 inputs
    // W = [[1, 0, 0],      b = [0, 0]
    //      [0, 1, 0]]
    // input = [3.0, 5.0, 7.0]
    // z[0] = 1*3 + 0*5 + 0*7 + 0 = 3.0
    // z[1] = 0*3 + 1*5 + 0*7 + 0 = 5.0
    Matrix<double> W(2, 3, {1,0,0,
                             0,1,0});
    Vector<double> b(2, 0.0);
    Layer<double, ReLU<double>> layer(std::move(W), std::move(b));

    Vector<double> input{3.0, 5.0, 7.0};
    const auto& a = layer.forward(input);

    EXPECT_NEAR(layer.z()[0], 3.0, kEps);
    EXPECT_NEAR(layer.z()[1], 5.0, kEps);
    EXPECT_NEAR(a[0], 3.0, kEps);  // ReLU(3) = 3
    EXPECT_NEAR(a[1], 5.0, kEps);  // ReLU(5) = 5
}

TEST(LayerForward, BiasContribution) {
    // Verify bias is added correctly
    // W = [[1, 0]], b = [10.0], input = [0.0]
    // z = 0 + 10 = 10,  a = ReLU(10) = 10
    Matrix<double> W(1, 1, {1.0});
    Vector<double> b{10.0};
    Layer<double, ReLU<double>> layer(std::move(W), std::move(b));

    const auto& a = layer.forward(Vector<double>{0.0});
    EXPECT_NEAR(a[0], 10.0, kEps);
}

TEST(LayerForward, OutputSizeMatchesFanOut) {
    Matrix<double> W(5, 3);
    Vector<double> b(5, 0.0);
    Layer<double, Sigmoid<double>> layer(std::move(W), std::move(b));

    Vector<double> input(3, 1.0);
    const auto& a = layer.forward(input);
    EXPECT_EQ(a.size(), 5u);
    EXPECT_EQ(layer.z().size(), 5u);
}

TEST(LayerForward, WrongInputSizeThrows) {
    Matrix<double> W(2, 3);
    Vector<double> b(2, 0.0);
    Layer<double, Sigmoid<double>> layer(std::move(W), std::move(b));

    Vector<double> wrong_input(2);
    EXPECT_THROW(layer.forward(wrong_input), std::invalid_argument);
}

TEST(LayerForward, ZPreActivationStoredCorrectly) {
    // z must be stored exactly as W*x + b, before activation
    // W = [[2, 3]], b = [-1], x = [1, 1]
    // z = 2 + 3 - 1 = 4, a = ReLU(4) = 4
    Matrix<double> W(1, 2, {2.0, 3.0});
    Vector<double> b{-1.0};
    Layer<double, ReLU<double>> layer(std::move(W), std::move(b));

    layer.forward(Vector<double>{1.0, 1.0});
    EXPECT_NEAR(layer.z()[0], 4.0, kEps);
    EXPECT_NEAR(layer.a()[0], 4.0, kEps);
}

// =========================================================================
// Xavier Initialization
// =========================================================================

TEST(LayerXavier, WeightsWithinLimit) {
    const std::size_t fan_in  = 64;
    const std::size_t fan_out = 32;
    const double limit = std::sqrt(6.0 / (fan_in + fan_out));

    Layer<double, Sigmoid<double>> layer(fan_in, fan_out, /*seed=*/42);

    for (std::size_t i = 0; i < fan_out; ++i) {
        for (std::size_t j = 0; j < fan_in; ++j) {
            EXPECT_GE(layer.W()(i, j), -limit)
                << "  weight (" << i << "," << j << ") below -limit";
            EXPECT_LE(layer.W()(i, j),  limit)
                << "  weight (" << i << "," << j << ") above +limit";
        }
    }
}

TEST(LayerXavier, BiasInitialisedToZero) {
    Layer<double, Sigmoid<double>> layer(4, 3, 42);
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(layer.b()[i], 0.0, kEps);
    }
}

TEST(LayerXavier, SeedReproducibility) {
    Layer<double, Sigmoid<double>> l1(8, 4, /*seed=*/123);
    Layer<double, Sigmoid<double>> l2(8, 4, /*seed=*/123);

    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 8; ++j)
            EXPECT_NEAR(l1.W()(i, j), l2.W()(i, j), kEps);
}

TEST(LayerXavier, DifferentSeedsDifferentWeights) {
    Layer<double, Sigmoid<double>> l1(8, 4, /*seed=*/1);
    Layer<double, Sigmoid<double>> l2(8, 4, /*seed=*/2);

    // Should differ in at least one weight
    bool any_different = false;
    for (std::size_t i = 0; i < 4 && !any_different; ++i)
        for (std::size_t j = 0; j < 8 && !any_different; ++j)
            if (std::abs(l1.W()(i, j) - l2.W()(i, j)) > kEps)
                any_different = true;

    EXPECT_TRUE(any_different);
}

TEST(LayerXavier, SmallLayerLimit) {
    // fan_in=2, fan_out=2 → limit = sqrt(6/4) = sqrt(1.5) ≈ 1.2247
    const double expected_limit = std::sqrt(6.0 / 4.0);
    Layer<double, Sigmoid<double>> layer(2, 2, 0);

    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            EXPECT_LE(std::abs(layer.W()(i, j)), expected_limit + kEps);
}

// =========================================================================
// Backward Pass — Numerical Gradient Check
// =========================================================================

// Compute ∂L/∂W[i,j] numerically via central difference on the loss.
// Loss = 0.5 * ||a - target||²  (simple quadratic, just for testing)
static double compute_loss(const Vector<double>& a, const Vector<double>& target) {
    double loss = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double diff = a[i] - target[i];
        loss += 0.5 * diff * diff;
    }
    return loss;
}

TEST(LayerBackward, GradientCheckW) {
    // Single-layer network: 2 inputs → 2 outputs (sigmoid)
    Matrix<double> W(2, 2, {0.3, -0.1,
                             0.2,  0.5});
    Vector<double> b{0.1, -0.2};
    Layer<double, Sigmoid<double>> layer(W, b);

    Vector<double> input{1.5, -0.5};
    Vector<double> target{1.0, 0.0};

    // Forward
    layer.forward(input);

    // Compute δ_out = a - target  (dL/da for quadratic loss = ŷ - y)
    Vector<double> delta_out = layer.a() - target;

    // Backward (output layer — delta provided directly)
    layer.backward_output(delta_out);

    const Matrix<double> grad_W_anal = layer.grad_W();

    // Numerical gradient via central difference
    const double h = 1e-5;
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            Matrix<double> W_fwd = W;
            Matrix<double> W_bwd = W;
            W_fwd(i, j) += h;
            W_bwd(i, j) -= h;

            Layer<double, Sigmoid<double>> l_fwd(W_fwd, b);
            Layer<double, Sigmoid<double>> l_bwd(W_bwd, b);

            const double loss_fwd = compute_loss(l_fwd.forward(input), target);
            const double loss_bwd = compute_loss(l_bwd.forward(input), target);

            const double grad_num = (loss_fwd - loss_bwd) / (2.0 * h);
            EXPECT_NEAR(grad_W_anal(i, j), grad_num, kGradEps)
                << "  grad_W[" << i << "," << j << "]";
        }
    }
}

TEST(LayerBackward, GradientCheckBias) {
    Matrix<double> W(2, 2, {0.3, -0.1,
                             0.2,  0.5});
    Vector<double> b{0.1, -0.2};
    Layer<double, Sigmoid<double>> layer(W, b);

    Vector<double> input{1.5, -0.5};
    Vector<double> target{1.0, 0.0};

    layer.forward(input);
    Vector<double> delta_out = layer.a() - target;
    layer.backward_output(delta_out);

    const Vector<double> grad_b_anal = layer.grad_b();

    const double h = 1e-5;
    for (std::size_t i = 0; i < 2; ++i) {
        Vector<double> b_fwd = b;
        Vector<double> b_bwd = b;
        b_fwd[i] += h;
        b_bwd[i] -= h;

        Layer<double, Sigmoid<double>> l_fwd(W, b_fwd);
        Layer<double, Sigmoid<double>> l_bwd(W, b_bwd);

        const double loss_fwd = compute_loss(l_fwd.forward(input), target);
        const double loss_bwd = compute_loss(l_bwd.forward(input), target);

        const double grad_num = (loss_fwd - loss_bwd) / (2.0 * h);
        EXPECT_NEAR(grad_b_anal[i], grad_num, kGradEps)
            << "  grad_b[" << i << "]";
    }
}

TEST(LayerBackward, HiddenLayerDeltaPropagation) {
    // Two-layer scenario: verify δ propagation from layer2 into layer1.
    //
    // Layer1: 2 inputs → 2 neurons (Sigmoid)
    // Layer2: 2 inputs → 1 neuron  (Sigmoid, output)
    //
    // We compute δ(layer1) = W2^T * δ(layer2) ⊙ σ'(z1)
    // and verify it matches the numerical gradient of layer1 weights.

    Matrix<double> W1(2, 2, { 0.3, -0.2,
                               0.1,  0.4});
    Vector<double> b1{0.0, 0.0};

    Matrix<double> W2(1, 2, {0.5, -0.3});
    Vector<double> b2{0.1};

    Layer<double, Sigmoid<double>> layer1(W1, b1);
    Layer<double, Sigmoid<double>> layer2(W2, b2);

    Vector<double> input{1.0, -1.0};
    Vector<double> target{1.0};

    // Forward
    const Vector<double>& a1 = layer1.forward(input);
    layer2.forward(a1);

    // Backward — output delta
    Vector<double> delta2_out = layer2.a() - target;
    layer2.backward_output(delta2_out);

    // Backward — hidden delta
    layer1.backward(layer2.delta(), layer2.W());

    const Matrix<double> grad_W1_anal = layer1.grad_W();

    // Numerical gradient for layer1 weights
    const double h = 1e-5;
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            Matrix<double> W1_fwd = W1;
            Matrix<double> W1_bwd = W1;
            W1_fwd(i, j) += h;
            W1_bwd(i, j) -= h;

            Layer<double, Sigmoid<double>> lf1(W1_fwd, b1);
            Layer<double, Sigmoid<double>> lf2(W2, b2);
            const double l_fwd = compute_loss(
                lf2.forward(lf1.forward(input)), target);

            Layer<double, Sigmoid<double>> lb1(W1_bwd, b1);
            Layer<double, Sigmoid<double>> lb2(W2, b2);
            const double l_bwd = compute_loss(
                lb2.forward(lb1.forward(input)), target);

            const double grad_num = (l_fwd - l_bwd) / (2.0 * h);
            EXPECT_NEAR(grad_W1_anal(i, j), grad_num, kGradEps)
                << "  hidden grad_W1[" << i << "," << j << "]";
        }
    }
}

TEST(LayerBackward, DeltaSizeMatchesFanOut) {
    Matrix<double> W(3, 2, {1,0, 0,1, 1,1});
    Vector<double> b(3, 0.0);
    Layer<double, Sigmoid<double>> layer(std::move(W), std::move(b));

    layer.forward(Vector<double>{1.0, 1.0});

    Vector<double> delta_next{0.1, 0.2, 0.3};
    Matrix<double> W_next(3, 3, 1.0);  // 3-output next layer (dummy)
    // Use backward_output instead for this size test
    layer.backward_output(delta_next);
    EXPECT_EQ(layer.delta().size(), 3u);
}

// =========================================================================
// SGD Update
// =========================================================================

TEST(LayerUpdate, WeightsChangeAfterUpdate) {
    Matrix<double> W(1, 2, {0.5, -0.3});
    Vector<double> b{0.1};
    Layer<double, Sigmoid<double>> layer(W, b);

    layer.forward(Vector<double>{1.0, 1.0});
    layer.backward_output(Vector<double>{0.5});

    const double old_w00 = layer.W()(0, 0);
    layer.update(0.01);

    EXPECT_NE(layer.W()(0, 0), old_w00);
}

TEST(LayerUpdate, GradientsZeroedAfterUpdate) {
    Matrix<double> W(1, 2, {0.5, -0.3});
    Vector<double> b{0.1};
    Layer<double, Sigmoid<double>> layer(W, b);

    layer.forward(Vector<double>{1.0, 1.0});
    layer.backward_output(Vector<double>{0.1});
    layer.update(0.01);

    for (std::size_t j = 0; j < 2; ++j)
        EXPECT_NEAR(layer.grad_W()(0, j), 0.0, kEps);
    EXPECT_NEAR(layer.grad_b()[0], 0.0, kEps);
}

// =========================================================================
// Template instantiation with different activations
// =========================================================================

TEST(LayerTemplate, WorksWithTanh) {
    Matrix<double> W(2, 2, {1,0, 0,1});
    Vector<double> b(2, 0.0);
    Layer<double, Tanh<double>> layer(std::move(W), std::move(b));

    const auto& a = layer.forward(Vector<double>{0.0, 0.0});
    // tanh(0) = 0
    EXPECT_NEAR(a[0], 0.0, kEps);
    EXPECT_NEAR(a[1], 0.0, kEps);
}

TEST(LayerTemplate, WorksWithReLU) {
    Matrix<double> W(2, 2, {1,0, 0,1});
    Vector<double> b(2, 0.0);
    Layer<double, ReLU<double>> layer(std::move(W), std::move(b));

    const auto& a = layer.forward(Vector<double>{-1.0, 2.0});
    EXPECT_NEAR(a[0], 0.0, kEps);  // ReLU(-1) = 0
    EXPECT_NEAR(a[1], 2.0, kEps);  // ReLU(2)  = 2
}
