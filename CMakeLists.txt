cmake_minimum_required(VERSION 3.19)

project(GPUReg LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

add_executable(GPUReg main_3d.cpp
    )

add_library(GPURegLib STATIC
    adamoptimiser.h adamoptimiser.cpp
    gpu.h gpu.cpp
    utils.h utils.cpp
    image.h image.cpp
    transform.h transform.cpp
)

add_library(nifti STATIC)
target_sources(nifti
    PRIVATE
        nifti/nifti1_io.c
        nifti/nifti1_io.h
        nifti/nifti1_io_version.h
        nifti/nifti1.h
        nifti/znzlib.c
        nifti/znzlib.h
        nifti/znzlib_version.h
)
target_include_directories(nifti
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/nifti
)

add_subdirectory(testing)


include(FetchContent)
FetchContent_Declare(
  webgpu
  GIT_REPOSITORY https://github.com/eliemichel/WebGPU-distribution
  GIT_TAG        dawn
)
FetchContent_MakeAvailable(webgpu)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/releases/download/v1.15.2/googletest-1.15.2.tar.gz
)
FetchContent_MakeAvailable(googletest)


FetchContent_Declare(matplotplusplus
        GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
        GIT_TAG origin/master) # or whatever tag you want
FetchContent_MakeAvailable(matplotplusplus)

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY "https://github.com/gabime/spdlog.git"
  GIT_TAG 8e5613379f5140fefb0b60412fbf1f5406e7c7f8 # v1.15.0
)
FetchContent_MakeAvailable(spdlog)

add_library(stb-image INTERFACE)
target_include_directories(stb-image INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/stb
)


target_link_libraries(GPURegLib PUBLIC
    webgpu_dawn
    stb-image
    spdlog::spdlog
    nifti
)

target_link_libraries(GPUReg PRIVATE GPURegLib matplot)

add_custom_command(TARGET GPUReg PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    "${PROJECT_SOURCE_DIR}/data"
    "$<TARGET_FILE_DIR:${PROJECT_NAME}>/data"
)

add_custom_target(shaders SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/downsample.wgsl
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/gradientx.wgsl
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/gradienty.wgsl
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/reduction.wgsl
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/reduction_f32.wgsl
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/transformimage.wgsl
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/updateparameters.wgsl
)

add_custom_command(TARGET GPUReg PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
    "$<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders"
)

include(GNUInstallDirs)
install(TARGETS GPUReg
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

