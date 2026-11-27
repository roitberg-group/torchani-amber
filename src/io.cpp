#include "io.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

#include <torch/all.h>

namespace {
// Open a file for appending, throws if it can't be opened
auto openForAppend(const std::string& path) -> std::ofstream {
    std::ofstream file(path, std::ios::app);
    if (!file) {
        throw std::runtime_error("Could not open " + path + " for writing");
    }
    return file;
}

// Same format as Fortran's F15.7
auto formatF15p7(double value) -> std::string {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%15.7f", value);
    return std::string(buffer);
}
}  // namespace

namespace io {

void createEmptyFile(const std::string& path) {
    std::ofstream file(path, std::ios::trunc);
    if (!file) {
        throw std::runtime_error("Could not create " + path);
    }
}

void appendXyzFrame(
    const std::string& path,
    const torch::Tensor& atomic_numbers,
    const torch::Tensor& coords,
    const std::string& comment
) {
    torch::Tensor atomic_numbers_cpu =
        atomic_numbers.to(torch::kCPU, torch::kLong).contiguous();
    torch::Tensor coords_cpu =
        coords.detach().to(torch::kCPU, torch::kDouble).contiguous();
    auto atomic_numbers_a = atomic_numbers_cpu.accessor<std::int64_t, 1>();
    auto coords_a = coords_cpu.accessor<double, 2>();
    std::ofstream file = openForAppend(path);
    char count[16];
    std::snprintf(
        count, sizeof(count), "%8lld", static_cast<long long>(coords_cpu.size(0))
    );
    file << count << "\n" << comment << "\n";
    for (std::int64_t atom = 0; atom != coords_cpu.size(0); ++atom) {
        char atomic_number[16];
        std::snprintf(
            atomic_number,
            sizeof(atomic_number),
            "%3lld",
            static_cast<long long>(atomic_numbers_a[atom])
        );
        file << atomic_number;
        for (int c = 0; c != 3; ++c) {
            file << formatF15p7(coords_a[atom][c]);
        }
        file << "\n";
    }
}

void appendDatFrame(const std::string& path, const torch::Tensor& values) {
    torch::Tensor values_cpu =
        values.detach().to(torch::kCPU, torch::kDouble).contiguous();
    if (values_cpu.dim() == 1) {
        values_cpu = values_cpu.unsqueeze(1);
    }
    auto values_a = values_cpu.accessor<double, 2>();
    std::ofstream file = openForAppend(path);
    for (std::int64_t row = 0; row != values_cpu.size(0); ++row) {
        for (std::int64_t col = 0; col != values_cpu.size(1); ++col) {
            file << formatF15p7(values_a[row][col]);
        }
        file << "\n";
    }
    file << "\n";
}

}  // namespace io
