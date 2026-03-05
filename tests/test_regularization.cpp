// tests/test_regularization.cpp
// Phase 11 — L2 Regularization tests.
//
// Tests:
//   L2Penalty.*         — compute_l2_penalty correctness and scaling
//   WeightDecay.*       — apply_weight_decay reduces weight norms
//   LossMonitoring.*    — epoch_losses include the regularization term
//   RegularizedTraining.* — training with L2 remains stable and shrinks weights
//   OverfitReduction.*  — L2 reduces weight magnitude vs unregularized training
//   Integration.*       — train_with_validation works with lambda > 0

#include "loss/l2_regularization.hpp"
#include "mlp/trainer.hpp"
#include "mlp/mlp.hpp"
#include "data/dataset.hpp"
#include "activations/sigmoid.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

// -----------------------------------------------------------------------
// Type aliases
// -----------------------------------------------------------------------
using F   = float;
using Act = mlp::Sigmoid<F>;
using TestMLP     = mlp::MLP<F, Act>;
using TestTrainer = mlp::Trainer<F, Act>;
using TestDataset = mlp::Dataset<F>;

// -----------------------------------------------------------------------
// Helper: build a dataset of n samples with 2 features and 2 classes.
// -----------------------------------------------------------------------
static TestDataset make_dataset(std::size_t n) {
    TestDataset ds;
    ds.inputs.reserve(n);
    ds.labels.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        mlp::Vector<F> x(2);
        x[0] = static_cast<F>(i) / static_cast<F>(n);
        x[1] = static_cast<F>(n - 1 - i) / static_cast<F>(n);
        ds.inputs.push_back(x);

        mlp::Vector<F> y(2, F{0});
        y[i % 2] = F{1};
        ds.labels.push_back(y);
    }
    return ds;
}

// -----------------------------------------------------------------------
// L2Penalty — compute_l2_penalty correctness
// -----------------------------------------------------------------------
TEST(L2Penalty, ZeroLambdaReturnsZero) {
    TestMLP mlp({2, 4, 2}, 42u);
    EXPECT_EQ(mlp::compute_l2_penalty(mlp, F{0}), F{0});
}

TEST(L2Penalty, PositiveForNonZeroWeights) {
    TestMLP mlp({2, 4, 2}, 42u);
    // Xavier-initialised weights are non-zero → penalty must be > 0
    const F penalty = mlp::compute_l2_penalty(mlp, F{1});
    EXPECT_GT(penalty, F{0});
}

TEST(L2Penalty, ScalesLinearlyWithLambda) {
    TestMLP mlp({2, 4, 2}, 42u);
    const F p1 = mlp::compute_l2_penalty(mlp, F{1});
    const F p2 = mlp::compute_l2_penalty(mlp, F{2});
    EXPECT_NEAR(p2, F{2} * p1, F{1e-5});
}

TEST(L2Penalty, MatchesManuallySummedWeights) {
    // Verify formula: lambda * Σ W²
    TestMLP mlp({2, 3, 2}, 99u);
    const F lambda = F{0.5};

    // Manually sum W² over all layers
    F manual_sum = F{0};
    for (std::size_t l = 0; l < mlp.num_layers(); ++l) {
        const auto& W = mlp.layer(l).W();
        for (std::size_t i = 0; i < W.rows(); ++i) {
            for (std::size_t j = 0; j < W.cols(); ++j) {
                manual_sum += W(i, j) * W(i, j);
            }
        }
    }
    const F expected = lambda * manual_sum;
    EXPECT_NEAR(mlp::compute_l2_penalty(mlp, lambda), expected, F{1e-5});
}

TEST(L2Penalty, WorksWithMultipleLayers) {
    // Deep network — penalty covers all layers
    TestMLP mlp({2, 8, 8, 4, 2}, 42u);
    const F penalty = mlp::compute_l2_penalty(mlp, F{1});
    EXPECT_GT(penalty, F{0});
    // 4 weight matrices → should be larger than single-layer penalty
    TestMLP mlp_shallow({2, 2}, 42u);
    EXPECT_GT(penalty, mlp::compute_l2_penalty(mlp_shallow, F{1}));
}

// -----------------------------------------------------------------------
// WeightDecay — apply_weight_decay reduces weight magnitudes
// -----------------------------------------------------------------------
TEST(WeightDecay, ReducesWeightNorm) {
    TestMLP mlp({2, 4, 2}, 42u);
    const F norm_before = mlp::compute_weight_norm_sq(mlp);

    mlp::apply_weight_decay(mlp, F{0.1});   // factor = 0.1 → W *= 0.9

    const F norm_after = mlp::compute_weight_norm_sq(mlp);
    EXPECT_LT(norm_after, norm_before);
}

TEST(WeightDecay, ZeroEtaLambdaIsNoOp) {
    TestMLP mlp({2, 4, 2}, 42u);
    const F norm_before = mlp::compute_weight_norm_sq(mlp);

    mlp::apply_weight_decay(mlp, F{0});

    const F norm_after = mlp::compute_weight_norm_sq(mlp);
    EXPECT_NEAR(norm_after, norm_before, F{1e-7});
}

TEST(WeightDecay, DecayFactorIsCorrect) {
    // With eta_lambda = 0.1: W_new = W * (1 - 0.1) = W * 0.9
    // norm_sq_new = 0.81 * norm_sq_old
    TestMLP mlp({2, 4, 2}, 42u);
    const F norm_sq_before = mlp::compute_weight_norm_sq(mlp);

    const F eta_lambda = F{0.1};
    mlp::apply_weight_decay(mlp, eta_lambda);

    const F norm_sq_after = mlp::compute_weight_norm_sq(mlp);
    const F factor_sq = (F{1} - eta_lambda) * (F{1} - eta_lambda);  // 0.81
    EXPECT_NEAR(norm_sq_after, factor_sq * norm_sq_before, F{1e-4});
}

// -----------------------------------------------------------------------
// LossMonitoring — epoch_losses include the regularization penalty
// -----------------------------------------------------------------------
TEST(LossMonitoring, RegularizedLossExceedsPureLoss) {
    // Same initial weights → same gradient step in epoch 1.
    // With lambda > 0 the reported loss includes L2 penalty → higher total.
    auto ds = make_dataset(6);

    TestMLP mlp_no_reg({2, 4, 2}, 42u);
    TestMLP mlp_with_reg({2, 4, 2}, 42u);   // identical initial state

    TestTrainer t_no_reg;
    TestTrainer t_with_reg;

    // Large lambda so the penalty dominates
    t_no_reg.train(mlp_no_reg,  ds, 1, F{0.01}, 1, false, 42u, F{0});
    t_with_reg.train(mlp_with_reg, ds, 1, F{0.01}, 1, false, 42u, F{2});

    EXPECT_GT(t_with_reg.epoch_losses()[0], t_no_reg.epoch_losses()[0])
        << "Regularized epoch loss should be larger (includes penalty)";
}

TEST(LossMonitoring, LossHistoryLengthUnchanged) {
    auto ds = make_dataset(6);
    TestMLP mlp({2, 4, 2}, 42u);
    TestTrainer trainer;

    const std::size_t epochs = 5;
    trainer.train(mlp, ds, epochs, F{0.01}, 1, false, 42u, F{0.1});

    EXPECT_EQ(trainer.epoch_losses().size(), epochs);
    EXPECT_EQ(trainer.epoch_accuracies().size(), epochs);
}

TEST(LossMonitoring, PenaltyIsFinite) {
    auto ds = make_dataset(6);
    TestMLP mlp({2, 4, 2}, 42u);
    TestTrainer trainer;

    trainer.train(mlp, ds, 10, F{0.01}, 1, false, 42u, F{0.1});

    for (F loss : trainer.epoch_losses()) {
        EXPECT_TRUE(std::isfinite(loss)) << "loss=" << loss;
        EXPECT_GE(loss, F{0});
    }
}

// -----------------------------------------------------------------------
// RegularizedTraining — L2 regularization shrinks weights over time
// -----------------------------------------------------------------------
TEST(RegularizedTraining, WeightsAreSmallerWithL2) {
    // Train two identical networks for many epochs.
    // The one with L2 should have smaller weight norms at the end.
    auto ds = make_dataset(8);

    TestMLP mlp_no_reg({2, 6, 2}, 42u);
    TestMLP mlp_with_reg({2, 6, 2}, 42u);

    TestTrainer t1, t2;

    const std::size_t epochs = 50;
    t1.train(mlp_no_reg,   ds, epochs, F{0.05}, 1, false, 42u, F{0.0});
    t2.train(mlp_with_reg, ds, epochs, F{0.05}, 1, false, 42u, F{0.5});

    const F norm_no_reg   = mlp::compute_weight_norm_sq(mlp_no_reg);
    const F norm_with_reg = mlp::compute_weight_norm_sq(mlp_with_reg);

    EXPECT_LT(norm_with_reg, norm_no_reg)
        << "L2-regularized model should have smaller weight norms";
}

TEST(RegularizedTraining, StableWithHighLambda) {
    // Verify that even with an aggressive lambda, training doesn't diverge.
    auto ds = make_dataset(6);
    TestMLP mlp({2, 4, 2}, 42u);
    TestTrainer trainer;

    trainer.train(mlp, ds, 20, F{0.01}, 1, false, 42u, F{1.0});

    // All weights should be finite after training
    for (std::size_t l = 0; l < mlp.num_layers(); ++l) {
        const auto& W = mlp.layer(l).W();
        for (std::size_t i = 0; i < W.rows(); ++i) {
            for (std::size_t j = 0; j < W.cols(); ++j) {
                EXPECT_TRUE(std::isfinite(W(i, j)))
                    << "weight NaN/Inf at layer=" << l
                    << " i=" << i << " j=" << j;
            }
        }
    }
}

// -----------------------------------------------------------------------
// OverfitReduction — L2 penalizes large weights (key property)
// -----------------------------------------------------------------------
TEST(OverfitReduction, LambdaZeroYieldsLargerWeights) {
    // A model trained without regularization on a small dataset
    // tends to grow larger weights than one with L2.
    auto ds = make_dataset(4);   // tiny dataset → prone to overfitting

    TestMLP mlp1({2, 8, 2}, 42u);
    TestMLP mlp2({2, 8, 2}, 42u);

    TestTrainer t1, t2;

    t1.train(mlp1, ds, 100, F{0.02}, 1, false, 42u, F{0.0});
    t2.train(mlp2, ds, 100, F{0.02}, 1, false, 42u, F{0.5});

    const F norm1 = mlp::compute_weight_norm_sq(mlp1);
    const F norm2 = mlp::compute_weight_norm_sq(mlp2);

    EXPECT_LT(norm2, norm1)
        << "Regularized model should have smaller weights "
           "(L2 penalizes large weights)";
}

TEST(OverfitReduction, LargerLambdaYieldsSmallerWeights) {
    auto ds = make_dataset(6);

    TestMLP mlp_low({2, 6, 2}, 42u);
    TestMLP mlp_high({2, 6, 2}, 42u);

    TestTrainer t1, t2;

    t1.train(mlp_low,  ds, 50, F{0.05}, 1, false, 42u, F{0.1});
    t2.train(mlp_high, ds, 50, F{0.05}, 1, false, 42u, F{1.0});

    EXPECT_LT(mlp::compute_weight_norm_sq(mlp_high),
              mlp::compute_weight_norm_sq(mlp_low))
        << "Higher lambda should produce smaller weights";
}

// -----------------------------------------------------------------------
// Integration — train_with_validation works correctly with lambda > 0
// -----------------------------------------------------------------------
TEST(Integration, TrainWithValidationWithL2) {
    auto ds = make_dataset(10);
    auto [train, val] = mlp::train_validation_split(ds, F{0.8}, 42u);

    TestMLP mlp({2, 4, 2}, 42u);
    TestTrainer trainer;

    mlp::EarlyStoppingConfig<F> es;
    es.patience  = 100;
    es.min_delta = F{0};

    // Should complete without exception
    EXPECT_NO_THROW(
        trainer.train_with_validation(mlp, train, val,
                                      /*epochs=*/10,
                                      F{0.01}, 1, false, 42u, es,
                                      /*lambda=*/F{0.1})
    );

    // History should be populated
    EXPECT_EQ(trainer.train_loss_history().size(), 10u);
    EXPECT_EQ(trainer.val_loss_history().size(),   10u);
}

TEST(Integration, TrainLossHigherThanValLossWithLargeReg) {
    // With lambda > 0, train_loss includes the L2 penalty.
    // val_loss is pure data loss.
    // For large lambda, train_loss should exceed val_loss at some epoch.
    auto ds = make_dataset(10);
    auto [train, val] = mlp::train_validation_split(ds, F{0.8}, 42u);

    TestMLP mlp({2, 4, 2}, 42u);
    TestTrainer trainer;

    mlp::EarlyStoppingConfig<F> es;
    es.patience  = 100;
    es.min_delta = F{0};

    trainer.train_with_validation(mlp, train, val,
                                  5, F{0.01}, 1, false, 42u, es,
                                  /*lambda=*/F{2.0});  // large penalty

    // At least one epoch should have train_loss > val_loss
    bool found = false;
    for (std::size_t e = 0; e < trainer.train_loss_history().size(); ++e) {
        if (trainer.train_loss_history()[e] > trainer.val_loss_history()[e]) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found)
        << "With large L2, regularized train_loss should exceed val_loss "
           "(val_loss has no penalty term)";
}

TEST(Integration, BackwardCompatibilityNoLambda) {
    // Ensure existing train() calls without lambda still work.
    auto ds = make_dataset(6);
    TestMLP mlp({2, 4, 2}, 42u);
    TestTrainer trainer;

    // This call uses the default lambda=0 (backward compatible)
    EXPECT_NO_THROW(trainer.train(mlp, ds, 3, F{0.01}));
    EXPECT_EQ(trainer.epoch_losses().size(), 3u);
}
