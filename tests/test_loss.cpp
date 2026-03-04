// tests/test_loss.cpp
// Phase 4 — Unit tests for mlp::CrossEntropy and mlp::MSE.

#include "loss/cross_entropy.hpp"
#include "loss/mse.hpp"
#include "core/vector.hpp"

#include <gtest/gtest.h>
#include <cmath>

using mlp::CrossEntropy;
using mlp::MSE;
using mlp::Vector;

static constexpr double kEps     = 1e-9;
static constexpr double kLossEps = 1e-7;  // tolerance for log-based comparisons

// =========================================================================
// CrossEntropy — compute_loss
// =========================================================================

TEST(CrossEntropyLoss, ManualThreeClasses) {
    // y     = [0, 1, 0]   (one-hot: class 1)
    // y_hat = [0.1, 0.8, 0.1]
    // L = -(0*log(0.1) + 1*log(0.8) + 0*log(0.1))
    //   = -log(0.8)
    //   ≈ 0.22314355
    CrossEntropy<double> ce;
    Vector<double> y_hat{0.1, 0.8, 0.1};
    Vector<double> y{0.0, 1.0, 0.0};

    const double expected = -std::log(0.8);
    EXPECT_NEAR(ce.compute_loss(y_hat, y), expected, kLossEps);
}

TEST(CrossEntropyLoss, PerfectPrediction) {
    // y_hat ≈ one-hot → loss ≈ 0
    CrossEntropy<double> ce;
    Vector<double> y_hat{0.0, 1.0, 0.0};
    Vector<double> y{0.0, 1.0, 0.0};

    // log(1.0) = 0 → loss = 0
    EXPECT_NEAR(ce.compute_loss(y_hat, y), 0.0, kLossEps);
}

TEST(CrossEntropyLoss, UniformPrediction) {
    // y_hat = [1/3, 1/3, 1/3],  y = [1, 0, 0]
    // L = -log(1/3) = log(3) ≈ 1.0986
    CrossEntropy<double> ce;
    Vector<double> y_hat{1.0/3.0, 1.0/3.0, 1.0/3.0};
    Vector<double> y{1.0, 0.0, 0.0};

    EXPECT_NEAR(ce.compute_loss(y_hat, y), std::log(3.0), kLossEps);
}

TEST(CrossEntropyLoss, TwoClasses) {
    // Binary case: y=[1,0], y_hat=[0.7, 0.3]
    // L = -log(0.7) ≈ 0.35667
    CrossEntropy<double> ce;
    Vector<double> y_hat{0.7, 0.3};
    Vector<double> y{1.0, 0.0};

    EXPECT_NEAR(ce.compute_loss(y_hat, y), -std::log(0.7), kLossEps);
}

TEST(CrossEntropyLoss, IsNonNegative) {
    CrossEntropy<double> ce;
    Vector<double> y_hat{0.2, 0.5, 0.3};
    Vector<double> y{0.0, 1.0, 0.0};
    EXPECT_GE(ce.compute_loss(y_hat, y), 0.0);
}

TEST(CrossEntropyLoss, SizeMismatchThrows) {
    CrossEntropy<double> ce;
    Vector<double> y_hat{0.5, 0.5};
    Vector<double> y{1.0, 0.0, 0.0};
    EXPECT_THROW({ auto r = ce.compute_loss(y_hat, y); (void)r; },
                 std::invalid_argument);
}

TEST(CrossEntropyLoss, EmptyVectorThrows) {
    CrossEntropy<double> ce;
    Vector<double> empty;
    EXPECT_THROW({ auto r = ce.compute_loss(empty, empty); (void)r; },
                 std::invalid_argument);
}

// =========================================================================
// CrossEntropy — Numerical Stability (no log(0) / NaN)
// =========================================================================

TEST(CrossEntropyStability, NearZeroProbability) {
    // y_hat[0] is very close to 0 but y[0]=0, so it's not used — no issue
    // y_hat[1] is near 0 but y[1]=1 — epsilon must protect log(0)
    CrossEntropy<double> ce;
    Vector<double> y_hat{0.9999, 1e-15, 1e-6};
    Vector<double> y{0.0, 1.0, 0.0};

    const double loss = ce.compute_loss(y_hat, y);
    EXPECT_TRUE(std::isfinite(loss)) << "loss is not finite: " << loss;
    EXPECT_GE(loss, 0.0);
    // Expected: -log(epsilon) ≈ -log(1e-12) ≈ 27.6
    EXPECT_GT(loss, 0.0);
}

TEST(CrossEntropyStability, ExactlyZeroProbabilityNotUsed) {
    // y[0]=1, y_hat[0]=0.9 — the zeros in y_hat not multiplied by nonzero y
    CrossEntropy<double> ce;
    Vector<double> y_hat{0.9, 0.0, 0.1};
    Vector<double> y{1.0, 0.0, 0.0};

    const double loss = ce.compute_loss(y_hat, y);
    EXPECT_TRUE(std::isfinite(loss));
    EXPECT_NEAR(loss, -std::log(0.9), kLossEps);
}

TEST(CrossEntropyStability, AllZerosPredictionClamped) {
    // y_hat[0]=0 with y[0]=1 — must clamp to epsilon
    CrossEntropy<double> ce;
    Vector<double> y_hat{0.0, 1.0};
    Vector<double> y{1.0, 0.0};

    const double loss = ce.compute_loss(y_hat, y);
    EXPECT_TRUE(std::isfinite(loss)) << "Expected finite loss, got: " << loss;
    EXPECT_GT(loss, 0.0);  // -log(epsilon) is a large positive number
}

TEST(CrossEntropyStability, NoNaNWithTinyProbabilities) {
    CrossEntropy<double> ce;
    Vector<double> y_hat{1e-300, 1.0 - 1e-300};
    Vector<double> y{1.0, 0.0};

    const double loss = ce.compute_loss(y_hat, y);
    EXPECT_FALSE(std::isnan(loss)) << "Got NaN";
    EXPECT_TRUE(std::isfinite(loss));
}

// =========================================================================
// CrossEntropy — compute_delta (Softmax + CE combined gradient)
// =========================================================================

TEST(CrossEntropyDelta, DeltaEqualsYHatMinusY) {
    CrossEntropy<double> ce;
    Vector<double> y_hat{0.1, 0.8, 0.1};
    Vector<double> y{0.0, 1.0, 0.0};

    Vector<double> delta = ce.compute_delta(y_hat, y);

    EXPECT_NEAR(delta[0],  0.1, kEps);   // 0.1 - 0
    EXPECT_NEAR(delta[1], -0.2, kEps);   // 0.8 - 1
    EXPECT_NEAR(delta[2],  0.1, kEps);   // 0.1 - 0
}

TEST(CrossEntropyDelta, PerfectPredictionDeltaZero) {
    CrossEntropy<double> ce;
    Vector<double> y_hat{0.0, 1.0, 0.0};
    Vector<double> y{0.0, 1.0, 0.0};

    Vector<double> delta = ce.compute_delta(y_hat, y);
    for (std::size_t i = 0; i < delta.size(); ++i) {
        EXPECT_NEAR(delta[i], 0.0, kEps) << "  at i=" << i;
    }
}

TEST(CrossEntropyDelta, SumDeltaEqualsZero) {
    // When y_hat is a valid probability distribution (sums to 1)
    // and y is one-hot (sums to 1), then Σ delta_i = Σ y_hat_i - Σ y_i = 0
    CrossEntropy<double> ce;
    Vector<double> y_hat{0.2, 0.5, 0.3};
    Vector<double> y{0.0, 1.0, 0.0};

    Vector<double> delta = ce.compute_delta(y_hat, y);
    double sum = 0.0;
    for (std::size_t i = 0; i < delta.size(); ++i) { sum += delta[i]; }
    EXPECT_NEAR(sum, 0.0, kEps);
}

TEST(CrossEntropyDelta, TwoClasses) {
    CrossEntropy<double> ce;
    Vector<double> y_hat{0.3, 0.7};
    Vector<double> y{0.0, 1.0};

    Vector<double> delta = ce.compute_delta(y_hat, y);
    EXPECT_NEAR(delta[0],  0.3, kEps);
    EXPECT_NEAR(delta[1], -0.3, kEps);
}

TEST(CrossEntropyDelta, SizeMismatchThrows) {
    CrossEntropy<double> ce;
    EXPECT_THROW(
        ce.compute_delta(Vector<double>{0.5, 0.5}, Vector<double>{1.0}),
        std::invalid_argument
    );
}

// =========================================================================
// CrossEntropy — float template instantiation
// =========================================================================

TEST(CrossEntropyTemplate, FloatType) {
    CrossEntropy<float> ce;
    Vector<float> y_hat{0.1f, 0.8f, 0.1f};
    Vector<float> y{0.0f, 1.0f, 0.0f};

    const float loss = ce.compute_loss(y_hat, y);
    EXPECT_TRUE(std::isfinite(loss));
    EXPECT_NEAR(static_cast<double>(loss), -std::log(0.8), 1e-5);
}

// =========================================================================
// MSE — compute_loss
// =========================================================================

TEST(MSELoss, ManualTwoElements) {
    // y=[0,1],  y_hat=[0.2, 0.8]
    // L = ((0.2-0)² + (0.8-1)²) / 2 = (0.04 + 0.04) / 2 = 0.04
    MSE<double> mse;
    Vector<double> y_hat{0.2, 0.8};
    Vector<double> y{0.0, 1.0};

    EXPECT_NEAR(mse.compute_loss(y_hat, y), 0.04, kLossEps);
}

TEST(MSELoss, PerfectPredictionIsZero) {
    MSE<double> mse;
    Vector<double> y{1.0, 0.0, 0.0};
    EXPECT_NEAR(mse.compute_loss(y, y), 0.0, kEps);
}

TEST(MSELoss, IsNonNegative) {
    MSE<double> mse;
    Vector<double> y_hat{0.1, 0.5, 0.4};
    Vector<double> y{1.0, 0.0, 0.0};
    EXPECT_GE(mse.compute_loss(y_hat, y), 0.0);
}

TEST(MSELoss, SymmetricInPredictionAndTarget) {
    // L(ŷ, y) == L(y, ŷ)
    MSE<double> mse;
    Vector<double> a{0.3, 0.7};
    Vector<double> b{0.6, 0.4};
    EXPECT_NEAR(mse.compute_loss(a, b), mse.compute_loss(b, a), kEps);
}

TEST(MSELoss, ScalesWithSquaredError) {
    // Doubling the error should quadruple the loss (per element)
    MSE<double> mse;
    Vector<double> y{0.0};
    Vector<double> y_hat1{1.0};
    Vector<double> y_hat2{2.0};

    const double l1 = mse.compute_loss(y_hat1, y);
    const double l2 = mse.compute_loss(y_hat2, y);
    EXPECT_NEAR(l2, 4.0 * l1, kEps);
}

TEST(MSELoss, SizeMismatchThrows) {
    MSE<double> mse;
    Vector<double> a{1.0, 2.0};
    Vector<double> b{1.0};
    EXPECT_THROW({ auto r = mse.compute_loss(a, b); (void)r; },
                 std::invalid_argument);
}

// =========================================================================
// MSE — compute_gradient
// =========================================================================

TEST(MSEGradient, ManualTwoElements) {
    // y=[0,1],  y_hat=[0.2, 0.8]
    // g = (2/2) * [0.2-0, 0.8-1] = [0.2, -0.2]
    MSE<double> mse;
    Vector<double> y_hat{0.2, 0.8};
    Vector<double> y{0.0, 1.0};

    Vector<double> g = mse.compute_gradient(y_hat, y);
    EXPECT_NEAR(g[0],  0.2, kEps);
    EXPECT_NEAR(g[1], -0.2, kEps);
}

TEST(MSEGradient, PerfectPredictionGradientIsZero) {
    MSE<double> mse;
    Vector<double> y{0.5, 0.5};
    Vector<double> g = mse.compute_gradient(y, y);
    for (std::size_t i = 0; i < g.size(); ++i) {
        EXPECT_NEAR(g[i], 0.0, kEps);
    }
}

TEST(MSEGradient, MatchesNumerical) {
    // Verify gradient numerically via central difference on the loss
    MSE<double> mse;
    Vector<double> y_hat{0.3, 0.5, 0.2};
    Vector<double> y{0.0, 1.0, 0.0};

    Vector<double> g_anal = mse.compute_gradient(y_hat, y);

    const double h = 1e-5;
    for (std::size_t i = 0; i < y_hat.size(); ++i) {
        Vector<double> yh_fwd = y_hat;
        Vector<double> yh_bwd = y_hat;
        yh_fwd[i] += h;
        yh_bwd[i] -= h;
        const double g_num =
            (mse.compute_loss(yh_fwd, y) - mse.compute_loss(yh_bwd, y)) /
            (2.0 * h);
        EXPECT_NEAR(g_anal[i], g_num, 1e-6) << "  at i=" << i;
    }
}

TEST(MSEGradient, SizeMismatchThrows) {
    MSE<double> mse;
    EXPECT_THROW(
        mse.compute_gradient(Vector<double>{1.0}, Vector<double>{1.0, 2.0}),
        std::invalid_argument
    );
}

// =========================================================================
// MSE — float template
// =========================================================================

TEST(MSETemplate, FloatType) {
    MSE<float> mse;
    Vector<float> y_hat{0.0f};
    Vector<float> y{1.0f};
    // L = (0-1)² / 1 = 1.0
    EXPECT_NEAR(static_cast<double>(mse.compute_loss(y_hat, y)), 1.0, 1e-6);
}
