// tests/test_vector.cpp
// Phase 1 — Unit tests for mlp::Vector<T>.

#include "core/vector.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <stdexcept>

using mlp::Vector;

// Epsilon for floating-point comparisons
static constexpr double kEps = 1e-9;

// ---- Construction -------------------------------------------------------

TEST(VectorConstruction, DefaultConstructor) {
    Vector<double> v;
    EXPECT_EQ(v.size(), 0u);
    EXPECT_TRUE(v.empty());
}

TEST(VectorConstruction, SizeConstructorZeroInit) {
    Vector<double> v(4);
    EXPECT_EQ(v.size(), 4u);
    for (std::size_t i = 0; i < v.size(); ++i) {
        EXPECT_NEAR(v[i], 0.0, kEps);
    }
}

TEST(VectorConstruction, SizeConstructorWithValue) {
    Vector<float> v(3, 5.0f);
    EXPECT_EQ(v.size(), 3u);
    for (std::size_t i = 0; i < v.size(); ++i) {
        EXPECT_NEAR(static_cast<double>(v[i]), 5.0, kEps);
    }
}

TEST(VectorConstruction, InitializerList) {
    Vector<double> v{1.0, 2.0, 3.0};
    EXPECT_EQ(v.size(), 3u);
    EXPECT_NEAR(v[0], 1.0, kEps);
    EXPECT_NEAR(v[1], 2.0, kEps);
    EXPECT_NEAR(v[2], 3.0, kEps);
}

TEST(VectorConstruction, IntegerType) {
    Vector<int> v{10, 20, 30};
    EXPECT_EQ(v[0], 10);
    EXPECT_EQ(v[1], 20);
    EXPECT_EQ(v[2], 30);
}

// ---- Accessors -----------------------------------------------------------

TEST(VectorAccessors, OperatorBracket) {
    Vector<double> v{3.0, 1.0, 4.0};
    EXPECT_NEAR(v[0], 3.0, kEps);
    EXPECT_NEAR(v[2], 4.0, kEps);
}

TEST(VectorAccessors, OperatorBracketMutate) {
    Vector<double> v(3, 0.0);
    v[1] = 7.5;
    EXPECT_NEAR(v[1], 7.5, kEps);
}

TEST(VectorAccessors, OutOfRangeThrows) {
    Vector<double> v(3);
    EXPECT_THROW(v[3], std::out_of_range);
    EXPECT_THROW(v[100], std::out_of_range);
}

TEST(VectorAccessors, ConstOutOfRangeThrows) {
    const Vector<double> v(2);
    EXPECT_THROW(v[2], std::out_of_range);
}

// ---- Sum -----------------------------------------------------------------

TEST(VectorSum, BasicAddition) {
    Vector<double> a{1.0, 2.0, 3.0};
    Vector<double> b{4.0, 5.0, 6.0};
    Vector<double> c = a + b;
    EXPECT_NEAR(c[0], 5.0, kEps);
    EXPECT_NEAR(c[1], 7.0, kEps);
    EXPECT_NEAR(c[2], 9.0, kEps);
}

TEST(VectorSum, AdditionWithNegatives) {
    Vector<double> a{1.0, -2.0, 3.0};
    Vector<double> b{-1.0, 2.0, -3.0};
    Vector<double> c = a + b;
    EXPECT_NEAR(c[0], 0.0, kEps);
    EXPECT_NEAR(c[1], 0.0, kEps);
    EXPECT_NEAR(c[2], 0.0, kEps);
}

TEST(VectorSum, AdditionSizeMismatchThrows) {
    Vector<double> a{1.0, 2.0};
    Vector<double> b{1.0, 2.0, 3.0};
    EXPECT_THROW(a + b, std::invalid_argument);
}

TEST(VectorSum, AddAssignOperator) {
    Vector<double> a{1.0, 2.0};
    Vector<double> b{3.0, 4.0};
    a += b;
    EXPECT_NEAR(a[0], 4.0, kEps);
    EXPECT_NEAR(a[1], 6.0, kEps);
}

// ---- Subtraction ---------------------------------------------------------

TEST(VectorSubtraction, Basic) {
    Vector<double> a{5.0, 3.0, 1.0};
    Vector<double> b{1.0, 1.0, 1.0};
    Vector<double> c = a - b;
    EXPECT_NEAR(c[0], 4.0, kEps);
    EXPECT_NEAR(c[1], 2.0, kEps);
    EXPECT_NEAR(c[2], 0.0, kEps);
}

TEST(VectorSubtraction, SelfSubtractionIsZero) {
    Vector<double> a{3.0, 5.0, 7.0};
    Vector<double> c = a - a;
    for (std::size_t i = 0; i < c.size(); ++i) {
        EXPECT_NEAR(c[i], 0.0, kEps);
    }
}

TEST(VectorSubtraction, SizeMismatchThrows) {
    Vector<double> a{1.0};
    Vector<double> b{1.0, 2.0};
    EXPECT_THROW(a - b, std::invalid_argument);
}

// ---- Dot Product ---------------------------------------------------------

TEST(VectorDot, BasicDotProduct) {
    // [1,2,3] · [4,5,6] = 4 + 10 + 18 = 32
    Vector<double> a{1.0, 2.0, 3.0};
    Vector<double> b{4.0, 5.0, 6.0};
    EXPECT_NEAR(a.dot(b), 32.0, kEps);
}

TEST(VectorDot, DotWithZeroVector) {
    Vector<double> a{1.0, 2.0, 3.0};
    Vector<double> zero(3, 0.0);
    EXPECT_NEAR(a.dot(zero), 0.0, kEps);
}

TEST(VectorDot, DotIsCommutative) {
    Vector<double> a{1.5, 2.5, 3.5};
    Vector<double> b{0.5, 1.5, 2.5};
    EXPECT_NEAR(a.dot(b), b.dot(a), kEps);
}

TEST(VectorDot, DotOrthogonalIsZero) {
    // [1,0] · [0,1] = 0
    Vector<double> a{1.0, 0.0};
    Vector<double> b{0.0, 1.0};
    EXPECT_NEAR(a.dot(b), 0.0, kEps);
}

TEST(VectorDot, SizeMismatchThrows) {
    Vector<double> a{1.0, 2.0};
    Vector<double> b{1.0, 2.0, 3.0};
    EXPECT_THROW({ auto r = a.dot(b); (void)r; }, std::invalid_argument);
}

// ---- Scalar Multiplication -----------------------------------------------

TEST(VectorScalar, VectorTimesScalar) {
    Vector<double> v{1.0, 2.0, 3.0};
    Vector<double> r = v * 3.0;
    EXPECT_NEAR(r[0], 3.0, kEps);
    EXPECT_NEAR(r[1], 6.0, kEps);
    EXPECT_NEAR(r[2], 9.0, kEps);
}

TEST(VectorScalar, ScalarTimesVector) {
    Vector<double> v{1.0, 2.0, 3.0};
    Vector<double> r = 2.0 * v;
    EXPECT_NEAR(r[0], 2.0, kEps);
    EXPECT_NEAR(r[1], 4.0, kEps);
    EXPECT_NEAR(r[2], 6.0, kEps);
}

TEST(VectorScalar, MultiplyByZero) {
    Vector<double> v{1.0, 2.0, 3.0};
    Vector<double> r = v * 0.0;
    for (std::size_t i = 0; i < r.size(); ++i) {
        EXPECT_NEAR(r[i], 0.0, kEps);
    }
}

TEST(VectorScalar, MultiplyByOne) {
    Vector<double> v{1.5, 2.5, 3.5};
    Vector<double> r = v * 1.0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        EXPECT_NEAR(r[i], v[i], kEps);
    }
}

// ---- Hadamard ------------------------------------------------------------

TEST(VectorHadamard, Basic) {
    Vector<double> a{2.0, 3.0, 4.0};
    Vector<double> b{5.0, 6.0, 7.0};
    Vector<double> r = a.hadamard(b);
    EXPECT_NEAR(r[0], 10.0, kEps);
    EXPECT_NEAR(r[1], 18.0, kEps);
    EXPECT_NEAR(r[2], 28.0, kEps);
}

// ---- Unary Negation ------------------------------------------------------

TEST(VectorNegation, Basic) {
    Vector<double> v{1.0, -2.0, 3.0};
    Vector<double> r = -v;
    EXPECT_NEAR(r[0], -1.0, kEps);
    EXPECT_NEAR(r[1],  2.0, kEps);
    EXPECT_NEAR(r[2], -3.0, kEps);
}

// ---- Numerical Precision -------------------------------------------------

TEST(VectorNumerical, FloatPrecision) {
    // Summing 0.1 ten times should be close to 1.0
    Vector<double> v(10, 0.1);
    double sum = 0.0;
    for (std::size_t i = 0; i < v.size(); ++i) { sum += v[i]; }
    EXPECT_NEAR(sum, 1.0, 1e-10);
}

TEST(VectorNumerical, LargeDotProduct) {
    // Known result: sum of squares 1..5 = 55
    Vector<double> v{1.0, 2.0, 3.0, 4.0, 5.0};
    EXPECT_NEAR(v.dot(v), 55.0, kEps);
}
