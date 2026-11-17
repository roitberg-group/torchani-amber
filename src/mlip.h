#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <torch/all.h>
#include <torch/script.h>

/**
 * Wrapper around a machine learning interatomic potential
 * */
class MlipModel {
    torch::jit::Module m_module;

    auto castFloatTensors(torch::ScalarType dtype) -> void;

    public:
    MlipModel() = default;
    MlipModel(const std::string& path, torch::Device device, torch::ScalarType dtype);

    auto useOnlyEnsembleMember(std::int64_t member_idx) -> void;
    auto readExtraFeaturesFromFile(const std::string& fpath) -> void;
    auto hasMethod(const std::string& name) const -> bool;
    auto supportsCustomMlmm() const -> bool;
    auto callMethod(const std::string& name, std::vector<torch::jit::IValue> inputs) const
        -> torch::jit::IValue;
    auto forward(std::vector<torch::jit::IValue> inputs) -> torch::jit::IValue;
    auto computeMlmm(std::vector<torch::jit::IValue> inputs) -> torch::jit::IValue;
};
