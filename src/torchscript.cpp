#include "torchscript.h"

#include <c10/util/Logging.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

namespace torchscript {

void disableTorchscriptExecutorOptimizations() {
    torch::jit::setGraphExecutorOptimize(false);
    LOG(INFO) << "Disabled TorchScript graph executor optimizations";
}

}  // namespace torchscript
