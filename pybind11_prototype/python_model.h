#pragma once

#include <string>
#include <utility>

#include <pybind11/pybind11.h>
#include <torch/torch.h>

// Only used by Python models
struct ModelOptions {
    bool skip_py_preamble = false;
    bool do_torch_compile = false;
    std::string torch_compile_mode = "default";
};

class PythonModel {
    public:
    PythonModel(
        const std::string& model_dir,
        torch::Device device,
        torch::ScalarType dtype,
        const ModelOptions& options = {}
    );
    auto computeEnergyAndForces(
        const torch::Tensor& atomic_numbers, const torch::Tensor& coords
    ) -> std::pair<torch::Tensor, torch::Tensor>;

    private:
    pybind11::object m_model;
};
