# tests/CMakeLists.txt

include(GoogleTest)

# ---- Phase 0: Infrastructure -----------------------------------------------
add_executable(test_placeholder
    test_placeholder.cpp
)
target_link_libraries(test_placeholder
    PRIVATE mlp_core GTest::gtest_main
)
gtest_discover_tests(test_placeholder)

# ---- Phase 1: Linear Algebra -----------------------------------------------
add_executable(test_vector
    test_vector.cpp
)
target_link_libraries(test_vector
    PRIVATE mlp_core GTest::gtest_main
)
gtest_discover_tests(test_vector)

add_executable(test_matrix
    test_matrix.cpp
)
target_link_libraries(test_matrix
    PRIVATE mlp_core GTest::gtest_main
)
gtest_discover_tests(test_matrix)

# ---- Phase 2: Activations --------------------------------------------------
add_executable(test_activations
    test_activations.cpp
)
target_link_libraries(test_activations
    PRIVATE mlp_core GTest::gtest_main
)
gtest_discover_tests(test_activations)

# ---- Phase 3: Layer --------------------------------------------------------
add_executable(test_layer
    test_layer.cpp
)
target_link_libraries(test_layer
    PRIVATE mlp_core GTest::gtest_main
)
gtest_discover_tests(test_layer)

# ---- Phase 4: Loss Functions -----------------------------------------------
add_executable(test_loss
    test_loss.cpp
)
target_link_libraries(test_loss
    PRIVATE mlp_core GTest::gtest_main
)
# Suppress GCC 13 false-positive -Wfree-nonheap-object triggered by
# certain RVO/NRVO patterns with std::vector inside template functions.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(test_loss PRIVATE -Wno-free-nonheap-object)
endif()
gtest_discover_tests(test_loss)

# ---- Phase 5: MLP ----------------------------------------------------------
add_executable(test_mlp
    test_mlp.cpp
)
target_link_libraries(test_mlp
    PRIVATE mlp_core GTest::gtest_main
)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(test_mlp PRIVATE -Wno-free-nonheap-object)
endif()
gtest_discover_tests(test_mlp)

# ---- Phase 6: Trainer ------------------------------------------------------
add_executable(test_training
    test_training.cpp
)
target_link_libraries(test_training
    PRIVATE mlp_core GTest::gtest_main
)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(test_training PRIVATE -Wno-free-nonheap-object)
endif()
gtest_discover_tests(test_training)

# ---- Phase 7: Data Pipeline ------------------------------------------------
add_executable(test_data
    test_data.cpp
)
target_link_libraries(test_data
    PRIVATE mlp_core GTest::gtest_main
)
# Inject project root so tests can locate train.csv at runtime.
target_compile_definitions(test_data
    PRIVATE TEST_DATA_DIR="${CMAKE_SOURCE_DIR}"
)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(test_data PRIVATE -Wno-free-nonheap-object)
endif()
gtest_discover_tests(test_data)

# ---- Phase 8: Metrics ------------------------------------------------------
add_executable(test_metrics
    test_metrics.cpp
)
target_link_libraries(test_metrics
    PRIVATE mlp_core GTest::gtest_main
)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(test_metrics PRIVATE -Wno-free-nonheap-object)
endif()
gtest_discover_tests(test_metrics)

# ---- Phase 9: Mini-batch Training ------------------------------------------
add_executable(test_minibatch
    test_minibatch.cpp
)
target_link_libraries(test_minibatch
    PRIVATE mlp_core GTest::gtest_main
)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(test_minibatch PRIVATE -Wno-free-nonheap-object)
endif()
gtest_discover_tests(test_minibatch)

# ---- Phase 10: Validation Split + Early Stopping ---------------------------
add_executable(test_validation
    test_validation.cpp
)
target_link_libraries(test_validation
    PRIVATE mlp_core GTest::gtest_main
)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(test_validation PRIVATE -Wno-free-nonheap-object)
endif()
gtest_discover_tests(test_validation)

# ---- Phase 11: L2 Regularization -------------------------------------------
add_executable(test_regularization
    test_regularization.cpp
)
target_link_libraries(test_regularization
    PRIVATE mlp_core GTest::gtest_main
)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(test_regularization PRIVATE -Wno-free-nonheap-object)
endif()
gtest_discover_tests(test_regularization)
