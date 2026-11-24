#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <torch/all.h>

#include "config.h"

namespace tensor_utils {

// Tensors stored by name. Storing a tensor with an existing name replaces it
class TensorContainer {
    std::unordered_map<std::string, torch::Tensor> m_tensors;

    public:
    auto store(const std::string& name, torch::Tensor tensor) -> void {
        m_tensors.insert_or_assign(name, std::move(tensor));
    }

    auto contains(const std::string& name) const -> bool {
        return m_tensors.count(name) > 0;
    }

    // Throws if there is no tensor with that name
    auto get(const std::string& name) const -> const torch::Tensor& {
        return m_tensors.at(name);
    }

    auto clear() -> void {
        m_tensors.clear();
    }

    // No-op if there is no tensor with that name
    auto clear(const std::string& name) -> void {
        m_tensors.erase(name);
    }
};

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
