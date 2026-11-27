#pragma once

#include <chrono>
#include <string>

namespace timing {

/**
 * Logs the time elapsed between construction and destruction with LOG(INFO)
 * */
class ScopedTimer {
    std::string m_label;
    std::chrono::steady_clock::time_point m_start;

    public:
    explicit ScopedTimer(std::string label);
    ScopedTimer(const ScopedTimer&) = delete;
    auto operator=(const ScopedTimer&) -> ScopedTimer& = delete;
    ~ScopedTimer();
};

}  // namespace timing
