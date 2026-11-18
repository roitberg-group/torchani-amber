#include "mlip.h"

#include <memory>
#include <string>

#include <torch/all.h>

#include "torchscript.h"

namespace mlip {

auto MlipModel::from_jit_model_path(
    const std::string& path, torch::Device device, torch::ScalarType dtype
) -> std::unique_ptr<MlipModel> {
    // Disable broken TorchScript "optimizations"
    torchscript::disableTorchscriptExecutorOptimizations();
    return std::make_unique<torchscript::TorchScriptMlipModel>(path, device, dtype);
}

}  // namespace mlip
