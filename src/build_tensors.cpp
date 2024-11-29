#include "build_tensors.h"
#include "config.h"

#include <vector>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/script.h>

// Torch needs as an input a TensorDataContainer to the
// torch::tensor constructor. The TensorDataContainer supports an initializer
// list, an at::ArrayRef or an std::vector of supported data types.
// An alternative used here to initialize from a pointer to a sequential
// memory location that has some data (from_blob).
// WARNING: It is crucial that
// the pointer type and the tensor type are the same when doing this,
// otherwise memory alignment issues occurr
// coords must be set to require grad, since automatic differentiation is
// used to get the forces
torch::Tensor dbl_buf_to_coords_tensor(
    const GlobalConfig& config,
    double coords_buf[][3],
    long num_atoms
) {
    // torchani needs an extra dimension as "batch dimension"
    return dbl_buf_to_tensor(config, coords_buf, {num_atoms, 3}).unsqueeze(0).requires_grad_(true);
}

torch::Tensor dbl_buf_to_shifts_tensor(
    const GlobalConfig& config,
    double shifts_buf[][3],
    long num_neighbors
) {
    return dbl_buf_to_tensor(config, shifts_buf, {num_neighbors, 3}).requires_grad_(true);
}

torch::Tensor dbl_buf_to_cell_tensor(
    const GlobalConfig& config,
    double cell_buf[][3]
) {
    torch::Tensor cell = dbl_buf_to_tensor(config, cell_buf, {3, 3});
    // Cell needs to be transposed due to ANI using opposite cell convention
#   ifdef DEBUG
    std::cout << "Torch cell:" << '\n'
              << cell << '\n'
              << "Transposed torch cell" << '\n'
              << torch::transpose(cell, 0, 1) << std::endl;
#   endif
    return torch::transpose(cell, 0, 1);
}

torch::Tensor int_buf_to_neighborlist_tensor(
    const GlobalConfig& config,
    int* neighborlist_buf[2],
    long num_neighbors
) {
    return int_buf_to_i64_tensor(config, neighborlist_buf, {2, num_neighbors});
}

torch::Tensor dbl_buf_to_tensor(
    const GlobalConfig& config,
    void* buf,
    const std::vector<long> shape_vec
) {
    torch::IntArrayRef shape{shape_vec.data(), shape_vec.size()};
    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kDouble);
    torch::Tensor tensor = torch::from_blob(buf, shape, options);
    return config.tensor_to_torchani_dtype_and_device(tensor);
}

torch::Tensor int_buf_to_i64_tensor(
    const GlobalConfig& config,
    void* buf,
    const std::vector<long> shape_vec
) {
    torch::IntArrayRef shape{shape_vec.data(), shape_vec.size()};
    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kInt);
    torch::Tensor tensor = torch::from_blob(buf, shape, options).to(torch::kLong);
    return config.tensor_to_torchani_device(tensor);
}
