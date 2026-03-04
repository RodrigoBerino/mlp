# src/CMakeLists.txt
# Each phase adds its headers to mlp_core so tests can include them uniformly.

add_library(mlp_core INTERFACE)
target_include_directories(mlp_core INTERFACE ${CMAKE_SOURCE_DIR}/src)
