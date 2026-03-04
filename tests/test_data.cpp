// tests/test_data.cpp
// Phase 7 — Tests for mlp::read_csv and mlp::DataPipeline.

#include "data/csv_reader.hpp"
#include "data/data_pipeline.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <fstream>
#include <numeric>
#include <string>

using mlp::CsvData;
using mlp::DataPipeline;
using mlp::Vector;
using mlp::read_csv;

// =========================================================================
// Fixture helpers
// =========================================================================

// Writes content to /tmp/mlp_test_fixture.csv and returns the path.
static std::string write_fixture(const std::string& content) {
    const std::string path = "/tmp/mlp_test_fixture.csv";
    std::ofstream f(path);
    f << content;
    return path;
}

// Small 6-row fixture: 4 numeric features, 1 string label (H / L / M).
static const char* kFixtureCsv =
    "raisedhands,VisITedResources,AnnouncementsView,Discussion,Class\n"
    "0,0,0,1,L\n"
    "50,50,50,50,M\n"
    "100,99,98,99,H\n"
    "10,20,30,40,L\n"
    "90,80,70,60,H\n"
    "40,40,40,40,M\n";

// Returns the path to the real Students dataset.
// TEST_DATA_DIR is injected by CMake as the project source root.
static std::string real_csv_path() {
    return std::string(TEST_DATA_DIR) + "/train.csv";
}

// =========================================================================
// CsvReader tests
// =========================================================================

TEST(CsvReader, ReadsHeaderCorrectly) {
    const std::string path = write_fixture(kFixtureCsv);
    CsvData csv = read_csv(path);

    ASSERT_EQ(csv.header.size(), 5u);
    EXPECT_EQ(csv.header[0], "raisedhands");
    EXPECT_EQ(csv.header[4], "Class");
}

TEST(CsvReader, ReadsCorrectRowCount) {
    const std::string path = write_fixture(kFixtureCsv);
    CsvData csv = read_csv(path);
    EXPECT_EQ(csv.rows.size(), 6u);
}

TEST(CsvReader, ParsesFieldValues) {
    const std::string path = write_fixture(kFixtureCsv);
    CsvData csv = read_csv(path);
    EXPECT_EQ(csv.rows[0][0], "0");
    EXPECT_EQ(csv.rows[0][4], "L");
    EXPECT_EQ(csv.rows[2][0], "100");
    EXPECT_EQ(csv.rows[2][4], "H");
}

TEST(CsvReader, FileNotFoundThrows) {
    EXPECT_THROW(read_csv("/tmp/mlp_no_such_file_xyz.csv"), std::runtime_error);
}

// =========================================================================
// DataPipeline — fit() and transform()
// =========================================================================

TEST(DataPipelineFit, LearnedMinMax) {
    DataPipeline<double> pipe;
    std::vector<std::vector<double>> rows = {
        {0.0, 10.0},
        {50.0, 30.0},
        {100.0, 20.0}
    };
    pipe.fit(rows);

    EXPECT_DOUBLE_EQ(pipe.feature_min()[0], 0.0);
    EXPECT_DOUBLE_EQ(pipe.feature_max()[0], 100.0);
    EXPECT_DOUBLE_EQ(pipe.feature_min()[1], 10.0);
    EXPECT_DOUBLE_EQ(pipe.feature_max()[1], 30.0);
}

TEST(DataPipelineFit, EmptyThrows) {
    DataPipeline<double> pipe;
    EXPECT_THROW(
        pipe.fit(std::vector<std::vector<double>>{}),
        std::invalid_argument);
}

TEST(DataPipelineTransform, ValuesInUnitInterval) {
    DataPipeline<double> pipe;
    pipe.fit({{0.0, 0.0}, {100.0, 100.0}});

    Vector<double> v = pipe.transform({50.0, 25.0});
    EXPECT_DOUBLE_EQ(v[0], 0.5);
    EXPECT_DOUBLE_EQ(v[1], 0.25);
}

TEST(DataPipelineTransform, ExtremaMapToZeroAndOne) {
    DataPipeline<double> pipe;
    pipe.fit({{0.0}, {100.0}});

    EXPECT_DOUBLE_EQ(pipe.transform({0.0})[0], 0.0);
    EXPECT_DOUBLE_EQ(pipe.transform({100.0})[0], 1.0);
}

TEST(DataPipelineTransform, ConstantFeatureYieldsZero) {
    DataPipeline<double> pipe;
    pipe.fit({{5.0}, {5.0}});   // max == min

    EXPECT_DOUBLE_EQ(pipe.transform({5.0})[0], 0.0);
}

// =========================================================================
// DataPipeline — encode()
// =========================================================================

TEST(DataPipelineEncode, AlphabeticalOrder) {
    // H < L < M  →  H=[1,0,0], L=[0,1,0], M=[0,0,1]
    DataPipeline<double> pipe;
    const std::string path = write_fixture(kFixtureCsv);
    pipe.load_and_split(path);

    ASSERT_EQ(pipe.class_names().size(), 3u);
    EXPECT_EQ(pipe.class_names()[0], "H");
    EXPECT_EQ(pipe.class_names()[1], "L");
    EXPECT_EQ(pipe.class_names()[2], "M");

    Vector<double> h = pipe.encode("H");
    EXPECT_DOUBLE_EQ(h[0], 1.0);
    EXPECT_DOUBLE_EQ(h[1], 0.0);
    EXPECT_DOUBLE_EQ(h[2], 0.0);

    Vector<double> m = pipe.encode("M");
    EXPECT_DOUBLE_EQ(m[0], 0.0);
    EXPECT_DOUBLE_EQ(m[1], 0.0);
    EXPECT_DOUBLE_EQ(m[2], 1.0);
}

TEST(DataPipelineEncode, UnknownLabelThrows) {
    DataPipeline<double> pipe;
    const std::string path = write_fixture(kFixtureCsv);
    pipe.load_and_split(path);
    EXPECT_THROW(pipe.encode("X"), std::invalid_argument);
}

TEST(DataPipelineEncode, OneHotExactlyOneBit) {
    DataPipeline<double> pipe;
    const std::string path = write_fixture(kFixtureCsv);
    pipe.load_and_split(path);

    for (const auto& cls : pipe.class_names()) {
        Vector<double> v = pipe.encode(cls);
        double sum = 0.0;
        for (std::size_t i = 0; i < v.size(); ++i) { sum += v[i]; }
        EXPECT_DOUBLE_EQ(sum, 1.0) << "class " << cls;
    }
}

// =========================================================================
// DataPipeline — load_and_split() sizes
// =========================================================================

TEST(DataPipelineSplit, SizesAddUp) {
    DataPipeline<double> pipe;
    const std::string path = write_fixture(kFixtureCsv);
    // 6 samples, 70%/15%/15%  →  train=4, val=0, test=2
    // (floor: 0.7*6=4, 0.15*6=0, 6-4-0=2)
    auto [train, val, test] = pipe.load_and_split(path);

    EXPECT_EQ(train.size() + val.size() + test.size(), 6u);
}

TEST(DataPipelineSplit, NoSampleLost) {
    DataPipeline<double> pipe;
    const std::string path = write_fixture(kFixtureCsv);
    auto [train, val, test] = pipe.load_and_split(path, /* last col */ std::numeric_limits<std::size_t>::max(), 0.5, 0.25);
    // 6 samples: train=3, val=1, test=2
    EXPECT_EQ(train.size() + val.size() + test.size(), 6u);
}

TEST(DataPipelineSplit, TrainFracPlusValFracGe1Throws) {
    DataPipeline<double> pipe;
    const std::string path = write_fixture(kFixtureCsv);
    EXPECT_THROW(
        pipe.load_and_split(path, std::numeric_limits<std::size_t>::max(), 0.7, 0.4),
        std::invalid_argument);
}

TEST(DataPipelineSplit, InputsNormalizedInUnitInterval) {
    DataPipeline<double> pipe;
    const std::string path = write_fixture(kFixtureCsv);
    auto [train, val, test] = pipe.load_and_split(path);

    auto check_dataset = [](const mlp::Dataset<double>& ds) {
        for (const auto& x : ds.inputs) {
            for (std::size_t j = 0; j < x.size(); ++j) {
                EXPECT_GE(x[j], 0.0) << "  feature " << j << " below 0";
                EXPECT_LE(x[j], 1.0) << "  feature " << j << " above 1";
            }
        }
    };
    check_dataset(train);
    check_dataset(val);
    check_dataset(test);
}

TEST(DataPipelineSplit, LabelsAreOneHot) {
    DataPipeline<double> pipe;
    const std::string path = write_fixture(kFixtureCsv);
    auto [train, val, test] = pipe.load_and_split(path);

    auto check_labels = [](const mlp::Dataset<double>& ds) {
        for (const auto& y : ds.labels) {
            double sum = 0.0;
            for (std::size_t i = 0; i < y.size(); ++i) { sum += y[i]; }
            EXPECT_NEAR(sum, 1.0, 1e-12);
        }
    };
    check_labels(train);
    check_labels(val);
    check_labels(test);
}

TEST(DataPipelineSplit, NumFeaturesAndClasses) {
    DataPipeline<double> pipe;
    const std::string path = write_fixture(kFixtureCsv);
    pipe.load_and_split(path);
    EXPECT_EQ(pipe.num_features(), 4u);
    EXPECT_EQ(pipe.num_classes(),  3u);
}

// =========================================================================
// DataPipeline — reproducibility and seed sensitivity
// =========================================================================

TEST(DataPipelineSplit, SameSeedSameOrder) {
    const std::string path = write_fixture(kFixtureCsv);

    DataPipeline<double> pipe1;
    auto [tr1, v1, te1] = pipe1.load_and_split(path, std::numeric_limits<std::size_t>::max(), 0.5, 0.25, 42);

    DataPipeline<double> pipe2;
    auto [tr2, v2, te2] = pipe2.load_and_split(path, std::numeric_limits<std::size_t>::max(), 0.5, 0.25, 42);

    ASSERT_EQ(tr1.size(), tr2.size());
    for (std::size_t i = 0; i < tr1.size(); ++i) {
        for (std::size_t j = 0; j < tr1.inputs[i].size(); ++j) {
            EXPECT_DOUBLE_EQ(tr1.inputs[i][j], tr2.inputs[i][j]);
        }
    }
}

TEST(DataPipelineSplit, DifferentSeedDifferentOrder) {
    const std::string path = write_fixture(kFixtureCsv);

    DataPipeline<double> pipe1;
    auto [tr1, v1, te1] = pipe1.load_and_split(path, std::numeric_limits<std::size_t>::max(), 0.5, 0.25, 1);

    DataPipeline<double> pipe2;
    auto [tr2, v2, te2] = pipe2.load_and_split(path, std::numeric_limits<std::size_t>::max(), 0.5, 0.25, 99);

    // At least the first sample of the training set should differ
    // for different seeds (probability of same order ≈ 1/720 for 6 items).
    bool any_different = false;
    for (std::size_t i = 0; i < tr1.size() && i < tr2.size(); ++i) {
        for (std::size_t j = 0; j < tr1.inputs[i].size(); ++j) {
            if (tr1.inputs[i][j] != tr2.inputs[i][j]) {
                any_different = true;
            }
        }
    }
    EXPECT_TRUE(any_different) << "Different seeds produced identical splits";
}

// =========================================================================
// Integration — real Students dataset
// =========================================================================

TEST(DataPipelineIntegration, LoadRealDataset) {
    DataPipeline<double> pipe;
    auto [train, val, test] = pipe.load_and_split(real_csv_path());

    // 336 samples total
    EXPECT_EQ(train.size() + val.size() + test.size(), 336u);

    // 4 features, 3 classes
    EXPECT_EQ(pipe.num_features(), 4u);
    EXPECT_EQ(pipe.num_classes(),  3u);

    // Class names alphabetical: H, L, M
    ASSERT_EQ(pipe.class_names().size(), 3u);
    EXPECT_EQ(pipe.class_names()[0], "H");
    EXPECT_EQ(pipe.class_names()[1], "L");
    EXPECT_EQ(pipe.class_names()[2], "M");
}

TEST(DataPipelineIntegration, TrainSplitSize) {
    DataPipeline<double> pipe;
    auto [train, val, test] = pipe.load_and_split(real_csv_path());
    // 70% of 336 = floor(235.2) = 235
    EXPECT_EQ(train.size(), 235u);
}

TEST(DataPipelineIntegration, FeaturesNormalizedAfterSplit) {
    DataPipeline<double> pipe;
    auto [train, val, test] = pipe.load_and_split(real_csv_path());

    for (const auto& x : train.inputs) {
        for (std::size_t j = 0; j < x.size(); ++j) {
            EXPECT_GE(x[j], 0.0);
            EXPECT_LE(x[j], 1.0);
        }
    }
}

TEST(DataPipelineIntegration, ReadyForTrainer) {
    // Confirm that the resulting Dataset can be used with Trainer without
    // throwing (i.e. inputs and labels have matching sizes and are valid).
    DataPipeline<double> pipe;
    auto [train, val, test] = pipe.load_and_split(real_csv_path());

    EXPECT_NO_THROW(train.validate());
    EXPECT_NO_THROW(val.validate());
    EXPECT_NO_THROW(test.validate());

    // Input size = 4, label size = 3
    ASSERT_FALSE(train.empty());
    EXPECT_EQ(train.inputs[0].size(), 4u);
    EXPECT_EQ(train.labels[0].size(), 3u);
}
