#pragma once

#include "spdlog/spdlog.h"
#include <chrono>
#include <string>


class ScopedTimer {
public:
    ScopedTimer(const std::string &name)
        : m_name(name), m_start(std::chrono::high_resolution_clock::now()) {}
    
    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - m_start);
        spdlog::info("{} took {} ms", m_name, duration.count());
    }
private:
    std::string m_name;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
};

