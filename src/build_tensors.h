#pragma once

#include <vector>

#include <torch/script.h>

#include "config.h"

/**
 * Utilities for building Tensors from buffers
 * */
torch::Tensor dblBufToTensor(
    const GlobalConfig& config,
    void* buf,
    const std::vector<long> shape_vec
);

torch::Tensor intBufToI64Tensor(
    const GlobalConfig& config,
    void* buf,
    const std::vector<long> shape_vec
);

torch::Tensor dblBufToCoordsTensor(
    const GlobalConfig& config,
    double coords_buf[][3],
    long num_atoms
);

torch::Tensor dblBufToShiftsTensor(
    const GlobalConfig& config,
    double shifts_buf[][3],
    long num_neighbors
);

torch::Tensor dblBufToCellTensor(
    const GlobalConfig& config,
    double cell_buf[][3]
);

torch::Tensor intBufToNeighborlistTensor(
    const GlobalConfig& config,
    int neighborlist_buf[],
    long num_neighbors
);
