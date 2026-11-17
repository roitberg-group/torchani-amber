#include "torchscript.h"

#include <ATen/Context.h>
#include <c10/util/Logging.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

void disableTorchscriptExecutorOptimizations() {
    torch::jit::setGraphExecutorOptimize(false);
    LOG(INFO) << "Disabled TorchScript graph executor optimizations";
}

void disableTorchReducedFloatPrecisionOptimizations() {
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
    at::globalContext().setAllowFP16ReductionCuBLAS(false);
    LOG(INFO) << "Disabled TF32 in cuBLAS and cuDNN, and FP16 reduction in cuBLAS";
}
