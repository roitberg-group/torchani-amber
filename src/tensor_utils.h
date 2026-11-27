#pragma once

#include <vector>

#include <torch/all.h>

#include "config.h"

namespace tensor_utils {

/**
 * Utilities for building Tensors from buffers
 * */
torch::Tensor dblBufToTensor(
    const config::GlobalConfig& config,
    void* buf,
    const std::vector<long> shape_vec
);

torch::Tensor intBufToI64Tensor(
    const config::GlobalConfig& config,
    void* buf,
    const std::vector<long> shape_vec
);

torch::Tensor dblBufToCoordsTensor(
    const config::GlobalConfig& config,
    double coords_buf[][3],
    long num_atoms
);

torch::Tensor dblBufToShiftsTensor(
    const config::GlobalConfig& config,
    double shifts_buf[][3],
    long num_neighbors
);

torch::Tensor dblBufToCellTensor(
    const config::GlobalConfig& config,
    double cell_buf[][3]
);

torch::Tensor intBufToNeighborlistTensor(
    const config::GlobalConfig& config,
    int neighborlist_buf[],
    long num_neighbors
);

}  // namespace tensor_utils
