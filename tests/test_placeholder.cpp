// tests/test_placeholder.cpp
// Phase 0: Infrastructure validation test.
// Verifies that the build system and test framework are correctly configured.

#include <gtest/gtest.h>

// Sanity check: confirms GoogleTest is linked and basic assertions work.
TEST(InfrastructureTest, BuildSystemIsConfigured) {
    EXPECT_TRUE(true);
}

// Confirms C++20 features are available.
TEST(InfrastructureTest, Cpp20IsEnabled) {
    // std::remove_cvref is a C++20 feature
    using T = std::remove_cvref_t<const int&>;
    EXPECT_TRUE((std::is_same_v<T, int>));
}

// Confirms compiler warning flags do not break compilation.
TEST(InfrastructureTest, CompilerFlagsAreCompatible) {
    constexpr int value = 42;
    EXPECT_EQ(value, 42);
}
