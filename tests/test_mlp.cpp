// tests/test_mlp.cpp
// Phase 5 — Integration tests for mlp::MLP<T, Activation>.

#include "mlp/mlp.hpp"
#include "activations/sigmoid.hpp"
#include "activations/relu.hpp"
#include "activations/tanh.hpp"
#include "loss/cross_entropy.hpp"
#include "core/vector.hpp"

#include <gtest/gtest.h>
#include <cmath>

using mlp::MLP;
using mlp::Vector;
using mlp::CrossEntropy;
using mlp::Sigmoid;
using mlp::ReLU;
using mlp::Tanh;

static constexpr double kEps     = 1e-9;
static constexpr double kGradEps = 1e-4;  // tolerance for gradient check

// =========================================================================
// Construction
// =========================================================================

TEST(MLPConstruction, LayerSizes) {
    MLP<double, Sigmoid<double>> mlp({2, 4, 3}, 42);
    EXPECT_EQ(mlp.num_layers(), 2u);
    EXPECT_EQ(mlp.layer(0).fan_in(),  2u);
    EXPECT_EQ(mlp.layer(0).fan_out(), 4u);
    EXPECT_EQ(mlp.layer(1).fan_in(),  4u);
    EXPECT_EQ(mlp.layer(1).fan_out(), 3u);
}

TEST(MLPConstruction, SingleLayerAllowed) {
    MLP<double, Sigmoid<double>> mlp({4, 3}, 42);
    EXPECT_EQ(mlp.num_layers(), 1u);
}

TEST(MLPConstruction, ThreeLayers) {
    MLP<double, Sigmoid<double>> mlp({2, 3, 4, 2}, 42);
    EXPECT_EQ(mlp.num_layers(), 3u);
}

TEST(MLPConstruction, TooFewSizesThrows) {
    EXPECT_THROW((MLP<double, Sigmoid<double>>({3}, 0)),
                 std::invalid_argument);
}

TEST(MLPConstruction, SeedReproducibility) {
    MLP<double, Sigmoid<double>> m1({2, 3, 2}, 42);
    MLP<double, Sigmoid<double>> m2({2, 3, 2}, 42);

    Vector<double> input{1.0, -1.0};
    const Vector<double>& out1 = m1.forward(input);
    const Vector<double>& out2 = m2.forward(input);

    for (std::size_t i = 0; i < out1.size(); ++i) {
        EXPECT_NEAR(out1[i], out2[i], kEps);
    }
}

// =========================================================================
// Forward Pass
// =========================================================================

TEST(MLPForward, OutputSizeMatchesLastLayer) {
    MLP<double, Sigmoid<double>> mlp({2, 3, 4}, 42);
    const auto& out = mlp.forward(Vector<double>{1.0, 0.5});
    EXPECT_EQ(out.size(), 4u);
}

TEST(MLPForward, SoftmaxOutputSumsToOne) {
    MLP<double, Sigmoid<double>> mlp({4, 6, 3}, 7);
    Vector<double> input{0.1, 0.2, 0.3, 0.4};
    const auto& out = mlp.forward(input);

    double sum = 0.0;
    for (std::size_t i = 0; i < out.size(); ++i) { sum += out[i]; }
    EXPECT_NEAR(sum, 1.0, 1e-9);
}

TEST(MLPForward, AllOutputProbabilitiesNonNegative) {
    MLP<double, Sigmoid<double>> mlp({3, 5, 3}, 99);
    const auto& out = mlp.forward(Vector<double>{-1.0, 0.0, 1.0});
    for (std::size_t i = 0; i < out.size(); ++i) {
        EXPECT_GE(out[i], 0.0);
    }
}

TEST(MLPForward, OutputIsFinite) {
    MLP<double, Sigmoid<double>> mlp({2, 4, 3}, 42);
    const auto& out = mlp.forward(Vector<double>{100.0, -100.0});
    for (std::size_t i = 0; i < out.size(); ++i) {
        EXPECT_TRUE(std::isfinite(out[i])) << "  non-finite at i=" << i;
    }
}

TEST(MLPForward, SingleLayerOutputSumsToOne) {
    MLP<double, Sigmoid<double>> mlp({3, 2}, 42);
    const auto& out = mlp.forward(Vector<double>{1.0, -1.0, 0.5});
    double sum = 0.0;
    for (std::size_t i = 0; i < out.size(); ++i) { sum += out[i]; }
    EXPECT_NEAR(sum, 1.0, kEps);
}

TEST(MLPForward, WrongInputSizeThrows) {
    MLP<double, Sigmoid<double>> mlp({3, 2}, 42);
    EXPECT_THROW(mlp.forward(Vector<double>{1.0, 2.0}),
                 std::invalid_argument);
}

// =========================================================================
// Loss
// =========================================================================

TEST(MLPLoss, IsNonNegative) {
    MLP<double, Sigmoid<double>> mlp({2, 4, 3}, 42);
    mlp.forward(Vector<double>{0.5, -0.5});
    EXPECT_GE(mlp.compute_loss(Vector<double>{0.0, 1.0, 0.0}), 0.0);
}

TEST(MLPLoss, PerfectPredictionIsNearZero) {
    // Force the output to be nearly one-hot by using an extreme logit
    // We can't guarantee exact one-hot due to Softmax, so just verify
    // that loss decreases as prediction improves.
    MLP<double, Sigmoid<double>> mlp({2, 3, 2}, 42);
    Vector<double> input{1.0, 0.0};
    Vector<double> y_true{0.0, 1.0};

    // Train for a while
    for (int step = 0; step < 500; ++step) {
        mlp.forward(input);
        mlp.backward(y_true);
        mlp.step(0.1);
    }
    const double loss = mlp.compute_loss(y_true);
    EXPECT_LT(loss, 0.3);
}

// =========================================================================
// Backpropagation — Numerical Gradient Check
// =========================================================================

// Full gradient check across all layers and all weights/biases.
// Strategy:
//   1. Run forward + backward → get analytical gradients (saved)
//   2. For each parameter, perturb, run forward only, compute CE loss
//   3. Compare analytical vs numerical gradient
//
// NOTE: gradients are saved BEFORE the perturbation loop, because
//       forward() does not touch grad_W/grad_b.

static void run_gradient_check(
    MLP<double, Sigmoid<double>>& mlp,
    const Vector<double>& input,
    const Vector<double>& y_true,
    double h       = 1e-5,
    double tol     = 1e-4)
{
    CrossEntropy<double> ce;

    // 1. Compute analytical gradients
    mlp.forward(input);
    mlp.backward(y_true);

    // 2. For each layer, save analytical gradients (copies!)
    const std::size_t L = mlp.num_layers();
    std::vector<mlp::Matrix<double>> grad_W_saved(L);
    std::vector<mlp::Vector<double>> grad_b_saved(L);
    for (std::size_t l = 0; l < L; ++l) {
        grad_W_saved[l] = mlp.layer(l).grad_W();
        grad_b_saved[l] = mlp.layer(l).grad_b();
    }

    // 3. Numerical check for every W[l][i][j]
    for (std::size_t l = 0; l < L; ++l) {
        const std::size_t rows = mlp.layer(l).W().rows();
        const std::size_t cols = mlp.layer(l).W().cols();

        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
                mlp.layer(l).W()(i, j) += h;
                const double lf = ce.compute_loss(mlp.forward(input), y_true);
                mlp.layer(l).W()(i, j) -= 2.0 * h;
                const double lb = ce.compute_loss(mlp.forward(input), y_true);
                mlp.layer(l).W()(i, j) += h;  // restore

                const double gnum  = (lf - lb) / (2.0 * h);
                const double ganal = grad_W_saved[l](i, j);
                EXPECT_NEAR(ganal, gnum, tol)
                    << "  Layer " << l
                    << "  W[" << i << "," << j << "]";
            }
        }

        // Numerical check for every b[l][i]
        for (std::size_t i = 0; i < mlp.layer(l).b().size(); ++i) {
            mlp.layer(l).b()[i] += h;
            const double lf = ce.compute_loss(mlp.forward(input), y_true);
            mlp.layer(l).b()[i] -= 2.0 * h;
            const double lb = ce.compute_loss(mlp.forward(input), y_true);
            mlp.layer(l).b()[i] += h;  // restore

            const double gnum  = (lf - lb) / (2.0 * h);
            const double ganal = grad_b_saved[l][i];
            EXPECT_NEAR(ganal, gnum, tol)
                << "  Layer " << l << "  b[" << i << "]";
        }
    }
}

TEST(MLPGradient, TwoLayerNetwork_2_3_2) {
    MLP<double, Sigmoid<double>> mlp({2, 3, 2}, 42);
    Vector<double> input{0.5, -0.3};
    Vector<double> y_true{1.0, 0.0};
    run_gradient_check(mlp, input, y_true);
}

TEST(MLPGradient, ThreeLayerNetwork_3_4_4_2) {
    MLP<double, Sigmoid<double>> mlp({3, 4, 4, 2}, 7);
    Vector<double> input{0.1, -0.2, 0.3};
    Vector<double> y_true{0.0, 1.0};
    run_gradient_check(mlp, input, y_true);
}

TEST(MLPGradient, SingleLayerNetwork_2_3) {
    MLP<double, Sigmoid<double>> mlp({2, 3}, 1);
    Vector<double> input{1.0, -1.0};
    Vector<double> y_true{0.0, 0.0, 1.0};
    run_gradient_check(mlp, input, y_true);
}

TEST(MLPGradient, SecondClassTarget) {
    MLP<double, Sigmoid<double>> mlp({2, 3, 2}, 13);
    Vector<double> input{-0.5, 0.7};
    Vector<double> y_true{0.0, 1.0};
    run_gradient_check(mlp, input, y_true);
}

// =========================================================================
// Training — Loss Decreases After One Step
// =========================================================================

TEST(MLPTraining, LossDecreasesAfterOneStep) {
    MLP<double, Sigmoid<double>> mlp({2, 4, 3}, 42);
    Vector<double> input{1.0, 0.5};
    Vector<double> y_true{0.0, 1.0, 0.0};

    mlp.forward(input);
    const double loss_before = mlp.compute_loss(y_true);
    mlp.backward(y_true);
    mlp.step(0.1);

    mlp.forward(input);
    const double loss_after = mlp.compute_loss(y_true);
    EXPECT_LT(loss_after, loss_before);
}

TEST(MLPTraining, LossDecreasesAfterMultipleSteps) {
    MLP<double, Sigmoid<double>> mlp({2, 4, 3}, 42);
    Vector<double> input{1.0, 0.5};
    Vector<double> y_true{0.0, 1.0, 0.0};

    mlp.forward(input);
    const double loss_initial = mlp.compute_loss(y_true);

    for (int s = 0; s < 50; ++s) {
        mlp.forward(input);
        mlp.backward(y_true);
        mlp.step(0.05);
    }

    mlp.forward(input);
    const double loss_final = mlp.compute_loss(y_true);
    EXPECT_LT(loss_final, loss_initial);
}

// =========================================================================
// Overfitting — Single Sample
// =========================================================================

TEST(MLPOverfit, SingleSampleConverges) {
    // A network should be able to memorise a single training sample.
    MLP<double, Sigmoid<double>> mlp({2, 8, 3}, 42);
    Vector<double> input{1.0, -1.0};
    Vector<double> y_true{0.0, 0.0, 1.0};  // class 2

    mlp.forward(input);
    const double loss_initial = mlp.compute_loss(y_true);

    for (int step = 0; step < 2000; ++step) {
        mlp.forward(input);
        mlp.backward(y_true);
        mlp.step(0.1);
    }

    mlp.forward(input);
    const double loss_final = mlp.compute_loss(y_true);

    EXPECT_LT(loss_final, loss_initial * 0.1)
        << "  loss did not decrease enough: "
        << loss_initial << " → " << loss_final;
    EXPECT_LT(loss_final, 0.5);
}

TEST(MLPOverfit, PredictedClassMatchesTarget) {
    // After overfitting, argmax of y_hat should equal the target class.
    MLP<double, Sigmoid<double>> mlp({2, 6, 3}, 42);
    Vector<double> input{0.5, 0.5};
    Vector<double> y_true{1.0, 0.0, 0.0};  // class 0

    for (int s = 0; s < 3000; ++s) {
        mlp.forward(input);
        mlp.backward(y_true);
        mlp.step(0.1);
    }

    const auto& out = mlp.forward(input);

    // Find argmax
    std::size_t predicted = 0;
    for (std::size_t i = 1; i < out.size(); ++i) {
        if (out[i] > out[predicted]) { predicted = i; }
    }
    EXPECT_EQ(predicted, 0u) << "  output: "
        << out[0] << " " << out[1] << " " << out[2];
}

// =========================================================================
// Numerical Stability
// =========================================================================

TEST(MLPNumerical, NoNaNAfterForward) {
    MLP<double, Sigmoid<double>> mlp({4, 8, 3}, 42);
    const auto& out = mlp.forward(Vector<double>{1e6, -1e6, 0.0, 1.0});
    for (std::size_t i = 0; i < out.size(); ++i) {
        EXPECT_FALSE(std::isnan(out[i])) << "  NaN at i=" << i;
    }
}

TEST(MLPNumerical, NoNaNAfterBackward) {
    MLP<double, Sigmoid<double>> mlp({3, 4, 2}, 42);
    Vector<double> input{0.1, 0.2, 0.3};
    Vector<double> y_true{1.0, 0.0};

    mlp.forward(input);
    mlp.backward(y_true);

    for (std::size_t l = 0; l < mlp.num_layers(); ++l) {
        for (std::size_t i = 0; i < mlp.layer(l).fan_out(); ++i) {
            for (std::size_t j = 0; j < mlp.layer(l).fan_in(); ++j) {
                EXPECT_FALSE(std::isnan(mlp.layer(l).grad_W()(i, j)))
                    << "  NaN in grad_W at layer=" << l;
            }
            EXPECT_FALSE(std::isnan(mlp.layer(l).grad_b()[i]))
                << "  NaN in grad_b at layer=" << l;
        }
    }
}

TEST(MLPNumerical, NoNaNAfterManySteps) {
    MLP<double, Sigmoid<double>> mlp({2, 4, 2}, 42);
    Vector<double> input{1.0, -1.0};
    Vector<double> y_true{0.0, 1.0};

    for (int s = 0; s < 500; ++s) {
        mlp.forward(input);
        mlp.backward(y_true);
        mlp.step(0.01);
    }

    const auto& out = mlp.forward(input);
    for (std::size_t i = 0; i < out.size(); ++i) {
        EXPECT_TRUE(std::isfinite(out[i]));
    }
}

// =========================================================================
// Template instantiation with different activations
// =========================================================================

TEST(MLPTemplate, WorksWithReLU) {
    MLP<double, ReLU<double>> mlp({2, 4, 3}, 42);
    Vector<double> input{1.0, 0.5};
    const auto& out = mlp.forward(input);

    double sum = 0.0;
    for (std::size_t i = 0; i < out.size(); ++i) { sum += out[i]; }
    EXPECT_NEAR(sum, 1.0, 1e-9);
}

TEST(MLPTemplate, WorksWithTanh) {
    MLP<double, Tanh<double>> mlp({2, 4, 3}, 42);
    Vector<double> input{0.5, -0.5};
    const auto& out = mlp.forward(input);

    double sum = 0.0;
    for (std::size_t i = 0; i < out.size(); ++i) { sum += out[i]; }
    EXPECT_NEAR(sum, 1.0, 1e-9);
}

TEST(MLPTemplate, GradientCheckWithTanh) {
    MLP<double, Tanh<double>> mlp({2, 3, 2}, 5);
    Vector<double> input{0.3, -0.7};
    Vector<double> y_true{1.0, 0.0};

    CrossEntropy<double> ce;

    mlp.forward(input);
    mlp.backward(y_true);

    // Save grad_W for output layer only
    mlp::Matrix<double> saved_grad = mlp.layer(1).grad_W();

    const double h = 1e-5;
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            mlp.layer(1).W()(i, j) += h;
            const double lf = ce.compute_loss(mlp.forward(input), y_true);
            mlp.layer(1).W()(i, j) -= 2.0 * h;
            const double lb = ce.compute_loss(mlp.forward(input), y_true);
            mlp.layer(1).W()(i, j) += h;

            EXPECT_NEAR(saved_grad(i, j), (lf - lb) / (2.0 * h), 1e-4)
                << "  Tanh grad_W[" << i << "," << j << "]";
        }
    }
}
