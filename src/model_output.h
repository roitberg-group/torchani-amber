#pragma once

#include <optional>

#include <torch/all.h>

#include "mlip.h"

// Shared by the model formats, uses no TorchScript
namespace model_output {

auto computeOutput(
    torch::Tensor energy,
    mlip::MlipOutput extra_outputs,
    const torch::Tensor& coords,
    const std::optional<torch::Tensor>& mm_coords,
    bool calc_forces,
    bool calc_qbc_and_qbc_grad
) -> mlip::MlipOutput;

}  // namespace model_output
