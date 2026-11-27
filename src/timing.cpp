#include "timing.h"

#include <chrono>
#include <string>
#include <utility>

#include <c10/util/Logging.h>

namespace timing {

ScopedTimer::ScopedTimer(std::string label)
    : m_label{std::move(label)}, m_start{std::chrono::steady_clock::now()} {}

ScopedTimer::~ScopedTimer() {
    std::chrono::duration<double, std::milli> elapsed =
        std::chrono::steady_clock::now() - m_start;
    LOG(INFO) << m_label << " time: " << elapsed.count() << " ms";
}

}  // namespace timing
