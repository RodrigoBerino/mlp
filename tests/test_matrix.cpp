// tests/test_matrix.cpp
// Phase 1 — Unit tests for mlp::Matrix<T>.

#include "core/matrix.hpp"
#include "core/vector.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <stdexcept>

using mlp::Matrix;
using mlp::Vector;

static constexpr double kEps = 1e-9;

// Helper to compare matrices element-wise with tolerance
static void expect_matrix_near(const Matrix<double>& A,
                                const Matrix<double>& B,
                                double eps = kEps) {
    ASSERT_EQ(A.rows(), B.rows());
    ASSERT_EQ(A.cols(), B.cols());
    for (std::size_t i = 0; i < A.rows(); ++i) {
        for (std::size_t j = 0; j < A.cols(); ++j) {
            EXPECT_NEAR(A(i, j), B(i, j), eps)
                << "  at (" << i << "," << j << ")";
        }
    }
}

// ---- Construction -------------------------------------------------------

TEST(MatrixConstruction, DefaultConstructor) {
    Matrix<double> m;
    EXPECT_EQ(m.rows(), 0u);
    EXPECT_EQ(m.cols(), 0u);
    EXPECT_TRUE(m.empty());
}

TEST(MatrixConstruction, SizeConstructorZeroInit) {
    Matrix<double> m(2, 3);
    EXPECT_EQ(m.rows(), 2u);
    EXPECT_EQ(m.cols(), 3u);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            EXPECT_NEAR(m(i, j), 0.0, kEps);
}

TEST(MatrixConstruction, SizeConstructorWithValue) {
    Matrix<double> m(3, 3, 1.0);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            EXPECT_NEAR(m(i, j), 1.0, kEps);
}

TEST(MatrixConstruction, InitializerList) {
    // 2x3 matrix: {{1,2,3},{4,5,6}}
    Matrix<double> m(2, 3, {1.0, 2.0, 3.0,
                             4.0, 5.0, 6.0});
    EXPECT_NEAR(m(0, 0), 1.0, kEps);
    EXPECT_NEAR(m(0, 2), 3.0, kEps);
    EXPECT_NEAR(m(1, 0), 4.0, kEps);
    EXPECT_NEAR(m(1, 2), 6.0, kEps);
}

TEST(MatrixConstruction, InitializerListSizeMismatchThrows) {
    EXPECT_THROW(
        (Matrix<double>(2, 2, {1.0, 2.0, 3.0})),
        std::invalid_argument
    );
}

TEST(MatrixConstruction, IntegerType) {
    Matrix<int> m(2, 2, {1, 2, 3, 4});
    EXPECT_EQ(m(0, 0), 1);
    EXPECT_EQ(m(1, 1), 4);
}

// ---- Accessors -----------------------------------------------------------

TEST(MatrixAccessors, OperatorParentheses) {
    Matrix<double> m(2, 2, {1.0, 2.0, 3.0, 4.0});
    EXPECT_NEAR(m(0, 0), 1.0, kEps);
    EXPECT_NEAR(m(0, 1), 2.0, kEps);
    EXPECT_NEAR(m(1, 0), 3.0, kEps);
    EXPECT_NEAR(m(1, 1), 4.0, kEps);
}

TEST(MatrixAccessors, OperatorParenthesesMutate) {
    Matrix<double> m(2, 2, 0.0);
    m(1, 0) = 9.9;
    EXPECT_NEAR(m(1, 0), 9.9, kEps);
}

TEST(MatrixAccessors, OutOfRangeThrows) {
    Matrix<double> m(2, 3);
    EXPECT_THROW(m(2, 0), std::out_of_range);  // row out of range
    EXPECT_THROW(m(0, 3), std::out_of_range);  // col out of range
}

// ---- Matrix-Vector Multiplication ----------------------------------------

TEST(MatrixVectorMul, IdentityTimesVector) {
    // I * v = v
    Matrix<double> I(3, 3, {1,0,0,
                             0,1,0,
                             0,0,1});
    Vector<double> v{2.0, 5.0, 7.0};
    Vector<double> r = I * v;
    EXPECT_NEAR(r[0], 2.0, kEps);
    EXPECT_NEAR(r[1], 5.0, kEps);
    EXPECT_NEAR(r[2], 7.0, kEps);
}

TEST(MatrixVectorMul, KnownResult) {
    // [[1,2],[3,4]] * [1,1] = [3,7]
    Matrix<double> A(2, 2, {1.0, 2.0,
                             3.0, 4.0});
    Vector<double> v{1.0, 1.0};
    Vector<double> r = A * v;
    EXPECT_NEAR(r[0], 3.0, kEps);
    EXPECT_NEAR(r[1], 7.0, kEps);
}

TEST(MatrixVectorMul, RectangularMatrix) {
    // [[1,0,2],[0,1,3]] * [1,2,3] = [7, 11]
    Matrix<double> A(2, 3, {1.0, 0.0, 2.0,
                             0.0, 1.0, 3.0});
    Vector<double> v{1.0, 2.0, 3.0};
    Vector<double> r = A * v;
    EXPECT_NEAR(r[0], 7.0, kEps);
    EXPECT_NEAR(r[1], 11.0, kEps);
}

TEST(MatrixVectorMul, SizeMismatchThrows) {
    Matrix<double> A(2, 3);
    Vector<double> v(2);
    EXPECT_THROW(A * v, std::invalid_argument);
}

// ---- Matrix-Matrix Multiplication ----------------------------------------

TEST(MatrixMatrixMul, IdentityTimesMatrix) {
    Matrix<double> I(3, 3, {1,0,0, 0,1,0, 0,0,1});
    Matrix<double> A(3, 3, {1,2,3, 4,5,6, 7,8,9});
    Matrix<double> R = I * A;
    expect_matrix_near(R, A);
}

TEST(MatrixMatrixMul, KnownResult2x2) {
    // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
    Matrix<double> A(2, 2, {1.0, 2.0,
                             3.0, 4.0});
    Matrix<double> B(2, 2, {5.0, 6.0,
                             7.0, 8.0});
    Matrix<double> C = A * B;
    EXPECT_NEAR(C(0, 0), 19.0, kEps);
    EXPECT_NEAR(C(0, 1), 22.0, kEps);
    EXPECT_NEAR(C(1, 0), 43.0, kEps);
    EXPECT_NEAR(C(1, 1), 50.0, kEps);
}

TEST(MatrixMatrixMul, RectangularKnownResult) {
    // A: 2x3, B: 3x2 → C: 2x2
    // A = [[1,2,3],[4,5,6]]
    // B = [[7,8],[9,10],[11,12]]
    // C[0,0] = 1*7 + 2*9 + 3*11 = 7+18+33 = 58
    // C[0,1] = 1*8 + 2*10 + 3*12 = 8+20+36 = 64
    // C[1,0] = 4*7 + 5*9 + 6*11 = 28+45+66 = 139
    // C[1,1] = 4*8 + 5*10 + 6*12 = 32+50+72 = 154
    Matrix<double> A(2, 3, {1,2,3, 4,5,6});
    Matrix<double> B(3, 2, {7,8, 9,10, 11,12});
    Matrix<double> C = A * B;
    ASSERT_EQ(C.rows(), 2u);
    ASSERT_EQ(C.cols(), 2u);
    EXPECT_NEAR(C(0, 0),  58.0, kEps);
    EXPECT_NEAR(C(0, 1),  64.0, kEps);
    EXPECT_NEAR(C(1, 0), 139.0, kEps);
    EXPECT_NEAR(C(1, 1), 154.0, kEps);
}

TEST(MatrixMatrixMul, SizeMismatchThrows) {
    Matrix<double> A(2, 3);
    Matrix<double> B(2, 2);
    EXPECT_THROW(A * B, std::invalid_argument);
}

// ---- Transpose -----------------------------------------------------------

TEST(MatrixTranspose, SquareMatrix) {
    // [[1,2],[3,4]]^T = [[1,3],[2,4]]
    Matrix<double> A(2, 2, {1.0, 2.0,
                             3.0, 4.0});
    Matrix<double> AT = A.transpose();
    EXPECT_NEAR(AT(0, 0), 1.0, kEps);
    EXPECT_NEAR(AT(0, 1), 3.0, kEps);
    EXPECT_NEAR(AT(1, 0), 2.0, kEps);
    EXPECT_NEAR(AT(1, 1), 4.0, kEps);
}

TEST(MatrixTranspose, RectangularMatrix) {
    // A: 2x3  →  AT: 3x2
    Matrix<double> A(2, 3, {1,2,3,
                             4,5,6});
    Matrix<double> AT = A.transpose();
    ASSERT_EQ(AT.rows(), 3u);
    ASSERT_EQ(AT.cols(), 2u);
    EXPECT_NEAR(AT(0, 0), 1.0, kEps);
    EXPECT_NEAR(AT(1, 0), 2.0, kEps);
    EXPECT_NEAR(AT(2, 0), 3.0, kEps);
    EXPECT_NEAR(AT(0, 1), 4.0, kEps);
    EXPECT_NEAR(AT(1, 1), 5.0, kEps);
    EXPECT_NEAR(AT(2, 1), 6.0, kEps);
}

TEST(MatrixTranspose, DoubleTransposeIsIdentity) {
    Matrix<double> A(2, 3, {1,2,3, 4,5,6});
    Matrix<double> ATA = A.transpose().transpose();
    expect_matrix_near(ATA, A);
}

TEST(MatrixTranspose, RowVector) {
    // 1x3 → 3x1
    Matrix<double> A(1, 3, {1.0, 2.0, 3.0});
    Matrix<double> AT = A.transpose();
    EXPECT_EQ(AT.rows(), 3u);
    EXPECT_EQ(AT.cols(), 1u);
    EXPECT_NEAR(AT(0, 0), 1.0, kEps);
    EXPECT_NEAR(AT(1, 0), 2.0, kEps);
    EXPECT_NEAR(AT(2, 0), 3.0, kEps);
}

// ---- Hadamard ------------------------------------------------------------

TEST(MatrixHadamard, Basic) {
    Matrix<double> A(2, 2, {1.0, 2.0, 3.0, 4.0});
    Matrix<double> B(2, 2, {5.0, 6.0, 7.0, 8.0});
    Matrix<double> R = A.hadamard(B);
    EXPECT_NEAR(R(0, 0),  5.0, kEps);
    EXPECT_NEAR(R(0, 1), 12.0, kEps);
    EXPECT_NEAR(R(1, 0), 21.0, kEps);
    EXPECT_NEAR(R(1, 1), 32.0, kEps);
}

TEST(MatrixHadamard, WithZeroMatrix) {
    Matrix<double> A(2, 2, {1.0, 2.0, 3.0, 4.0});
    Matrix<double> Z(2, 2, 0.0);
    Matrix<double> R = A.hadamard(Z);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            EXPECT_NEAR(R(i, j), 0.0, kEps);
}

TEST(MatrixHadamard, ShapeMismatchThrows) {
    Matrix<double> A(2, 2);
    Matrix<double> B(2, 3);
    EXPECT_THROW(A.hadamard(B), std::invalid_argument);
}

// ---- Arithmetic ----------------------------------------------------------

TEST(MatrixArithmetic, AdditionKnownResult) {
    Matrix<double> A(2, 2, {1,2,3,4});
    Matrix<double> B(2, 2, {5,6,7,8});
    Matrix<double> C = A + B;
    EXPECT_NEAR(C(0, 0), 6.0,  kEps);
    EXPECT_NEAR(C(0, 1), 8.0,  kEps);
    EXPECT_NEAR(C(1, 0), 10.0, kEps);
    EXPECT_NEAR(C(1, 1), 12.0, kEps);
}

TEST(MatrixArithmetic, SubtractionKnownResult) {
    Matrix<double> A(2, 2, {5,6,7,8});
    Matrix<double> B(2, 2, {1,2,3,4});
    Matrix<double> C = A - B;
    EXPECT_NEAR(C(0, 0), 4.0, kEps);
    EXPECT_NEAR(C(0, 1), 4.0, kEps);
    EXPECT_NEAR(C(1, 0), 4.0, kEps);
    EXPECT_NEAR(C(1, 1), 4.0, kEps);
}

TEST(MatrixArithmetic, ScalarMultiplication) {
    Matrix<double> A(2, 2, {1,2,3,4});
    Matrix<double> R = A * 2.0;
    EXPECT_NEAR(R(0, 0), 2.0, kEps);
    EXPECT_NEAR(R(0, 1), 4.0, kEps);
    EXPECT_NEAR(R(1, 0), 6.0, kEps);
    EXPECT_NEAR(R(1, 1), 8.0, kEps);
}

TEST(MatrixArithmetic, ScalarMultiplicationCommutative) {
    Matrix<double> A(2, 2, {1,2,3,4});
    Matrix<double> R1 = A * 3.0;
    Matrix<double> R2 = 3.0 * A;
    expect_matrix_near(R1, R2);
}

// ---- Numerical Precision -------------------------------------------------

TEST(MatrixNumerical, MultiplicationPrecision) {
    // For a 2x2 known result, check floating point accuracy
    // [[0.1, 0.2],[0.3, 0.4]] * [[10,0],[0,10]] = [[1,2],[3,4]]
    Matrix<double> A(2, 2, {0.1, 0.2, 0.3, 0.4});
    Matrix<double> B(2, 2, {10.0, 0.0, 0.0, 10.0});
    Matrix<double> C = A * B;
    EXPECT_NEAR(C(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(C(0, 1), 2.0, 1e-9);
    EXPECT_NEAR(C(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(C(1, 1), 4.0, 1e-9);
}

TEST(MatrixNumerical, TransposeMulAssociativity) {
    // (AB)^T == B^T A^T
    Matrix<double> A(2, 3, {1,2,3, 4,5,6});
    Matrix<double> B(3, 2, {7,8, 9,10, 11,12});
    Matrix<double> AB  = (A * B).transpose();
    Matrix<double> BtAt = B.transpose() * A.transpose();
    expect_matrix_near(AB, BtAt, 1e-9);
}
