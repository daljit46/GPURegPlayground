cmake_minimum_required(VERSION 3.5)

project(GPURegPlayground LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

add_executable(GPURegPlayground main.cpp
    gpu.h gpu.cpp
    utils.h utils.cpp
    image.h
    shaders/updategradients.wgsl
)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)
FetchContent_Declare(
  webgpu
  GIT_REPOSITORY https://github.com/eliemichel/WebGPU-distribution
  GIT_TAG        dawn
)
FetchContent_MakeAvailable(webgpu)

FetchContent_Declare(
    matplotplusplus
    GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
    GIT_TAG        master
)
FetchContent_MakeAvailable(matplotplusplus)

add_library(stb-image INTERFACE)
target_include_directories(stb-image INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/stb
)

target_link_libraries(GPURegPlayground PRIVATE
    webgpu_dawn
    stb-image
    matplot
)

add_custom_command(TARGET GPURegPlayground PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    "${CMAKE_CURRENT_SOURCE_DIR}/data"
    "$<TARGET_FILE_DIR:${PROJECT_NAME}>/data"
)

add_custom_target(shaders SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/gradientx.wgsl
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/gradienty.wgsl
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/transformimage.wgsl
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/updategradients.wgsl
)

add_custom_command(TARGET GPURegPlayground PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
    "$<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders"
)


include(GNUInstallDirs)
install(TARGETS GPURegPlayground
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

