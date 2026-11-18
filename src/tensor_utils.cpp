#include "tensor_utils.h"

#include <vector>

#include <torch/all.h>

#include "config.h"

namespace tensor_utils {

// Torch needs as an input a TensorDataContainer to the
// torch::tensor constructor. The TensorDataContainer supports an initializer
// list, an at::ArrayRef or an std::vector of supported data types.
// An alternative used here to initialize from a pointer to a sequential
// memory location that has some data (from_blob).
// WARNING: It is crucial that
// the pointer type and the tensor type are the same when doing this,
// otherwise memory alignment issues occur
// coords must be set to require grad, since automatic differentiation is
// used to get the forces
torch::Tensor dblBufToCoordsTensor(
    const config::GlobalConfig& config,
    double coords_buf[][3],
    long num_atoms
) {
    // torchani needs an extra dimension as "batch dimension"
    return dblBufToTensor(config, coords_buf, {num_atoms, 3}).unsqueeze(0).requires_grad_(true);
}

torch::Tensor dblBufToShiftsTensor(
    const config::GlobalConfig& config,
    double shifts_buf[][3],
    long num_neighbors
) {
    return dblBufToTensor(config, shifts_buf, {num_neighbors, 3}).requires_grad_(true);
}

torch::Tensor dblBufToCellTensor(
    const config::GlobalConfig& config,
    double cell_buf[][3]
) {
    return dblBufToTensor(config, cell_buf, {3, 3});
}

torch::Tensor intBufToNeighborlistTensor(
    const config::GlobalConfig& config,
    int neighborlist_buf[],
    long num_neighbors
) {
    return intBufToI64Tensor(config, neighborlist_buf, {2, num_neighbors});
}

torch::Tensor dblBufToTensor(
    const config::GlobalConfig& config,
    void* buf,
    const std::vector<long> shape_vec
) {
    torch::IntArrayRef shape{shape_vec.data(), shape_vec.size()};
    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kDouble);
    torch::Tensor tensor = torch::from_blob(buf, shape, options);
    return config.tensorToTorchaniDtypeAndDevice(tensor);
}

torch::Tensor intBufToI64Tensor(
    const config::GlobalConfig& config,
    void* buf,
    const std::vector<long> shape_vec
) {
    torch::IntArrayRef shape{shape_vec.data(), shape_vec.size()};
    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kInt);
    torch::Tensor tensor = torch::from_blob(buf, shape, options).to(torch::kLong);
    return config.tensorToTorchaniDevice(tensor);
}

}  // namespace tensor_utils
