// tests/test_activations.cpp
// Phase 2 — Unit tests for activation functors.
// Covers: Sigmoid, Tanh, ReLU, Softmax
// Includes: forward values, derivatives, numerical stability.

#include "activations/sigmoid.hpp"
#include "activations/tanh.hpp"
#include "activations/relu.hpp"
#include "activations/softmax.hpp"
#include "core/vector.hpp"

#include <gtest/gtest.h>
#include <cmath>

using mlp::Sigmoid;
using mlp::Tanh;
using mlp::ReLU;
using mlp::Softmax;
using mlp::Vector;

static constexpr double kEps    = 1e-9;   // exact comparisons
static constexpr double kDerEps = 1e-5;   // numerical derivative tolerance

// Helper: numerical derivative via central difference
// f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
template<typename Func, typename T>
static T numerical_derivative(Func&& f, T x, T h = T{1e-5}) {
    return (f(x + h) - f(x - h)) / (T{2} * h);
}

// =========================================================================
// Sigmoid
// =========================================================================

TEST(SigmoidForward, AtZero) {
    Sigmoid<double> s;
    EXPECT_NEAR(s(0.0), 0.5, kEps);
}

TEST(SigmoidForward, PositiveInput) {
    Sigmoid<double> s;
    // σ(1) = 1/(1+e^{-1}) ≈ 0.7310585786
    EXPECT_NEAR(s(1.0), 1.0 / (1.0 + std::exp(-1.0)), kEps);
}

TEST(SigmoidForward, NegativeInput) {
    Sigmoid<double> s;
    EXPECT_NEAR(s(-1.0), 1.0 / (1.0 + std::exp(1.0)), kEps);
}

TEST(SigmoidForward, LargePositive) {
    Sigmoid<double> s;
    // σ(+large) → 1
    EXPECT_NEAR(s(100.0), 1.0, 1e-9);
}

TEST(SigmoidForward, LargeNegative) {
    Sigmoid<double> s;
    // σ(-large) → 0
    EXPECT_NEAR(s(-100.0), 0.0, 1e-9);
}

TEST(SigmoidForward, SymmetryProperty) {
    Sigmoid<double> s;
    // σ(x) + σ(-x) = 1
    EXPECT_NEAR(s(2.0) + s(-2.0), 1.0, kEps);
}

TEST(SigmoidDerivative, AtZero) {
    Sigmoid<double> s;
    // σ'(0) = 0.5 * 0.5 = 0.25
    EXPECT_NEAR(s.derivative(0.0), 0.25, kEps);
}

TEST(SigmoidDerivative, MatchesNumerical) {
    Sigmoid<double> s;
    for (double x : {-2.0, -1.0, 0.0, 1.0, 2.0}) {
        const double analytical = s.derivative(x);
        const double numerical  = numerical_derivative([&](double v){ return s(v); }, x);
        EXPECT_NEAR(analytical, numerical, kDerEps)
            << "  at x=" << x;
    }
}

TEST(SigmoidDerivative, IsNonNegative) {
    Sigmoid<double> s;
    for (double x : {-5.0, -2.0, 0.0, 2.0, 5.0}) {
        EXPECT_GE(s.derivative(x), 0.0);
    }
}

// =========================================================================
// Tanh
// =========================================================================

TEST(TanhForward, AtZero) {
    Tanh<double> t;
    EXPECT_NEAR(t(0.0), 0.0, kEps);
}

TEST(TanhForward, PositiveInput) {
    Tanh<double> t;
    EXPECT_NEAR(t(1.0), std::tanh(1.0), kEps);
}

TEST(TanhForward, NegativeInput) {
    Tanh<double> t;
    EXPECT_NEAR(t(-1.0), std::tanh(-1.0), kEps);
}

TEST(TanhForward, OddFunction) {
    Tanh<double> t;
    // tanh is odd: tanh(-x) = -tanh(x)
    EXPECT_NEAR(t(-2.0), -t(2.0), kEps);
}

TEST(TanhForward, BoundedOutput) {
    Tanh<double> t;
    // tanh ∈ [-1, 1]; at extreme inputs IEEE 754 saturates to ±1.0 exactly
    EXPECT_GE(t(100.0),  -1.0);
    EXPECT_LE(t(100.0),   1.0);
    EXPECT_NEAR(t(100.0),   1.0, 1e-9);
    EXPECT_NEAR(t(-100.0), -1.0, 1e-9);
}

TEST(TanhDerivative, AtZero) {
    Tanh<double> t;
    // tanh'(0) = 1 - 0² = 1
    EXPECT_NEAR(t.derivative(0.0), 1.0, kEps);
}

TEST(TanhDerivative, MatchesNumerical) {
    Tanh<double> t;
    for (double x : {-2.0, -1.0, 0.0, 1.0, 2.0}) {
        const double analytical = t.derivative(x);
        const double numerical  = numerical_derivative([&](double v){ return t(v); }, x);
        EXPECT_NEAR(analytical, numerical, kDerEps)
            << "  at x=" << x;
    }
}

TEST(TanhDerivative, InRange) {
    Tanh<double> t;
    // tanh'(x) ∈ (0, 1]
    for (double x : {-5.0, -1.0, 0.0, 1.0, 5.0}) {
        EXPECT_GT(t.derivative(x), 0.0);
        EXPECT_LE(t.derivative(x), 1.0);
    }
}

// =========================================================================
// ReLU
// =========================================================================

TEST(ReLUForward, PositiveInput) {
    ReLU<double> r;
    EXPECT_NEAR(r(3.5), 3.5, kEps);
}

TEST(ReLUForward, NegativeInput) {
    ReLU<double> r;
    EXPECT_NEAR(r(-3.5), 0.0, kEps);
}

TEST(ReLUForward, AtZero) {
    ReLU<double> r;
    EXPECT_NEAR(r(0.0), 0.0, kEps);
}

TEST(ReLUForward, LargeValues) {
    ReLU<double> r;
    EXPECT_NEAR(r(1e6), 1e6, kEps);
    EXPECT_NEAR(r(-1e6), 0.0, kEps);
}

TEST(ReLUDerivative, PositiveInput) {
    ReLU<double> r;
    EXPECT_NEAR(r.derivative(1.0), 1.0, kEps);
    EXPECT_NEAR(r.derivative(0.5), 1.0, kEps);
}

TEST(ReLUDerivative, NegativeInput) {
    ReLU<double> r;
    EXPECT_NEAR(r.derivative(-1.0), 0.0, kEps);
}

TEST(ReLUDerivative, AtZeroIsZero) {
    ReLU<double> r;
    // Subgradient convention: derivative(0) = 0
    EXPECT_NEAR(r.derivative(0.0), 0.0, kEps);
}

TEST(ReLUDerivative, MatchesNumericalPositive) {
    ReLU<double> r;
    for (double x : {0.5, 1.0, 2.0, 5.0}) {
        const double analytical = r.derivative(x);
        const double numerical  = numerical_derivative([&](double v){ return r(v); }, x);
        EXPECT_NEAR(analytical, numerical, kDerEps)
            << "  at x=" << x;
    }
}

// =========================================================================
// Softmax
// =========================================================================

TEST(SoftmaxForward, SumIsOne) {
    Softmax<double> sm;
    Vector<double> z{1.0, 2.0, 3.0};
    Vector<double> p = sm(z);
    double sum = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i) { sum += p[i]; }
    EXPECT_NEAR(sum, 1.0, kEps);
}

TEST(SoftmaxForward, AllProbabilitiesNonNegative) {
    Softmax<double> sm;
    Vector<double> z{-1.0, 0.0, 1.0, 2.0};
    Vector<double> p = sm(z);
    for (std::size_t i = 0; i < p.size(); ++i) {
        EXPECT_GE(p[i], 0.0) << "  at i=" << i;
    }
}

TEST(SoftmaxForward, UniformInputGivesUniformOutput) {
    Softmax<double> sm;
    Vector<double> z{2.0, 2.0, 2.0, 2.0};
    Vector<double> p = sm(z);
    for (std::size_t i = 0; i < p.size(); ++i) {
        EXPECT_NEAR(p[i], 0.25, kEps) << "  at i=" << i;
    }
}

TEST(SoftmaxForward, OrderPreserved) {
    Softmax<double> sm;
    // Larger logit → larger probability
    Vector<double> z{1.0, 3.0, 2.0};
    Vector<double> p = sm(z);
    EXPECT_GT(p[1], p[2]);
    EXPECT_GT(p[2], p[0]);
}

TEST(SoftmaxForward, KnownResult) {
    Softmax<double> sm;
    // z = [0, 1]
    // p[0] = e^0/(e^0+e^1) = 1/(1+e)
    // p[1] = e^1/(e^0+e^1) = e/(1+e)
    Vector<double> z{0.0, 1.0};
    Vector<double> p = sm(z);
    const double denom = 1.0 + std::exp(1.0);
    EXPECT_NEAR(p[0], 1.0 / denom, kEps);
    EXPECT_NEAR(p[1], std::exp(1.0) / denom, kEps);
}

TEST(SoftmaxForward, TranslationInvariant) {
    Softmax<double> sm;
    // softmax(z) == softmax(z + c) for any constant c
    Vector<double> z{1.0, 2.0, 3.0};
    Vector<double> z_shift{101.0, 102.0, 103.0};
    Vector<double> p1 = sm(z);
    Vector<double> p2 = sm(z_shift);
    for (std::size_t i = 0; i < p1.size(); ++i) {
        EXPECT_NEAR(p1[i], p2[i], kEps) << "  at i=" << i;
    }
}

// ---- Numerical stability -------------------------------------------------

TEST(SoftmaxStability, LargeInputsNoOverflow) {
    Softmax<double> sm;
    // Without max-subtraction these would produce inf/nan
    Vector<double> z{1000.0, 1001.0, 1002.0};
    Vector<double> p = sm(z);

    // Check no NaN or Inf
    for (std::size_t i = 0; i < p.size(); ++i) {
        EXPECT_TRUE(std::isfinite(p[i])) << "  non-finite at i=" << i;
        EXPECT_GE(p[i], 0.0);
    }

    // Sum must still be 1
    double sum = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i) { sum += p[i]; }
    EXPECT_NEAR(sum, 1.0, 1e-9);

    // Largest logit must have largest probability
    EXPECT_GT(p[2], p[1]);
    EXPECT_GT(p[1], p[0]);
}

TEST(SoftmaxStability, VeryNegativeInputs) {
    Softmax<double> sm;
    Vector<double> z{-1000.0, -1001.0, -1002.0};
    Vector<double> p = sm(z);

    for (std::size_t i = 0; i < p.size(); ++i) {
        EXPECT_TRUE(std::isfinite(p[i]));
        EXPECT_GE(p[i], 0.0);
    }

    double sum = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i) { sum += p[i]; }
    EXPECT_NEAR(sum, 1.0, 1e-9);
}

TEST(SoftmaxStability, MixedLargeSmall) {
    Softmax<double> sm;
    Vector<double> z{-500.0, 0.0, 500.0};
    Vector<double> p = sm(z);

    for (std::size_t i = 0; i < p.size(); ++i) {
        EXPECT_TRUE(std::isfinite(p[i]));
    }

    double sum = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i) { sum += p[i]; }
    EXPECT_NEAR(sum, 1.0, 1e-9);

    // The 500.0 entry should absorb almost all probability
    EXPECT_NEAR(p[2], 1.0, 1e-9);
}

TEST(SoftmaxStability, SingleElement) {
    Softmax<double> sm;
    Vector<double> z{42.0};
    Vector<double> p = sm(z);
    EXPECT_NEAR(p[0], 1.0, kEps);
}

TEST(SoftmaxStability, EmptyInputThrows) {
    Softmax<double> sm;
    Vector<double> z;
    EXPECT_THROW(sm(z), std::invalid_argument);
}

// ---- Jacobian-vector product ---------------------------------------------

TEST(SoftmaxJacobian, SumsToZero) {
    Softmax<double> sm;
    // (J·1)_i = s_i*(1 - Σs_j) = 0 since Σs_j=1
    Vector<double> z{1.0, 2.0, 3.0};
    Vector<double> ones(3, 1.0);
    Vector<double> jv = sm.jacobian_times_vec(z, ones);
    double total = 0.0;
    for (std::size_t i = 0; i < jv.size(); ++i) { total += jv[i]; }
    EXPECT_NEAR(total, 0.0, kDerEps);
}

TEST(SoftmaxJacobian, MatchesNumerical) {
    Softmax<double> sm;
    Vector<double> z{0.5, 1.5, -0.5};
    Vector<double> v{1.0, 0.0, 0.0};   // probe first column of J

    // Analytical Jv
    Vector<double> jv_anal = sm.jacobian_times_vec(z, v);

    // Numerical: (sm(z+hv) - sm(z-hv)) / 2h  element-wise
    const double h = 1e-5;
    Vector<double> z_fwd{z[0]+h, z[1], z[2]};
    Vector<double> z_bwd{z[0]-h, z[1], z[2]};
    Vector<double> p_fwd = sm(z_fwd);
    Vector<double> p_bwd = sm(z_bwd);

    for (std::size_t i = 0; i < z.size(); ++i) {
        const double numerical = (p_fwd[i] - p_bwd[i]) / (2.0 * h);
        EXPECT_NEAR(jv_anal[i], numerical, kDerEps)
            << "  at i=" << i;
    }
}

// =========================================================================
// Float type (template instantiation check)
// =========================================================================

TEST(ActivationTemplateFloat, SigmoidFloat) {
    Sigmoid<float> s;
    EXPECT_NEAR(static_cast<double>(s(0.0f)), 0.5, 1e-6);
}

TEST(ActivationTemplateFloat, ReLUFloat) {
    ReLU<float> r;
    EXPECT_NEAR(static_cast<double>(r(2.0f)), 2.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(r(-1.0f)), 0.0, 1e-6);
}

TEST(ActivationTemplateFloat, SoftmaxFloat) {
    Softmax<float> sm;
    Vector<float> z{1.0f, 2.0f, 3.0f};
    Vector<float> p = sm(z);
    float sum = 0.0f;
    for (std::size_t i = 0; i < p.size(); ++i) { sum += p[i]; }
    EXPECT_NEAR(static_cast<double>(sum), 1.0, 1e-6);
}
