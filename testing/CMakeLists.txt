# Hello
enable_testing()

add_executable(GPURegTest
    gputests.cpp
    shadertests.cpp
)

add_custom_command(TARGET GPURegTest PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    "${PROJECT_SOURCE_DIR}/data"
    "${CMAKE_CURRENT_BINARY_DIR}/data"
)

add_custom_command(TARGET GPURegTest PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    "${PROJECT_SOURCE_DIR}/shaders"
    "${CMAKE_CURRENT_BINARY_DIR}/shaders"
)

target_link_libraries(GPURegTest PRIVATE
    GTest::gtest_main
    GPURegLib
)

include(GoogleTest)
gtest_discover_tests(GPURegTest)
