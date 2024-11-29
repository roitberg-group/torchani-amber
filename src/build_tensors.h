#pragma once

#include "config.h"

#include <vector>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/script.h>

/**
 * Utilities for building Tensors from buffers
 * */
torch::Tensor dbl_buf_to_tensor(
    const GlobalConfig& config,
    void* buf,
    const std::vector<long> shape_vec
);

torch::Tensor int_buf_to_i64_tensor(
    const GlobalConfig& config,
    void* buf,
    const std::vector<long> shape_vec
);

torch::Tensor dbl_buf_to_coords_tensor(
    const GlobalConfig& config,
    double coords_buf[][3],
    long num_atoms
);

torch::Tensor dbl_buf_to_shifts_tensor(
    const GlobalConfig& config,
    double shifts_buf[][3],
    long num_neighbors
);

torch::Tensor dbl_buf_to_cell_tensor(
    const GlobalConfig& config,
    double cell_buf[][3]
);

torch::Tensor int_buf_to_neighborlist_tensor(
    const GlobalConfig& config,
    int* neighborlist_buf[2],
    long num_neighbors
);
