cmake_minimum_required(VERSION 3.5)

project(GPURegPlayground LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

add_executable(GPURegPlayground main.cpp
    wgpucontext.h wgpucontext.cpp)

include(dawn)

target_link_libraries(GPURegPlayground PRIVATE libdawn)

include(GNUInstallDirs)
install(TARGETS GPURegPlayground
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

