#pragma once

#include <string>

#include <torch/all.h>

namespace io {

// Create an empty file at path, replacing it if it exists
void createEmptyFile(const std::string& path);

// Append a frame to an .xyz trajectory file. atomic_numbers has shape (atoms,) and
// coords has shape (atoms, 3)
void appendXyzFrame(
    const std::string& path,
    const torch::Tensor& atomic_numbers,
    const torch::Tensor& coords,
    const std::string& comment
);

// Append values to a .dat file, one row per atom, followed by an empty line. values
// has shape (atoms,) or (atoms, columns)
void appendDatFrame(const std::string& path, const torch::Tensor& values);

}  // namespace io
