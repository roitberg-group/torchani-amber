#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <torch/all.h>
#include <torch/script.h>

/**
 * Wrapper around a JIT-compiled machine learning interatomic potential
 * */
class MlipModel {
    torch::jit::Module m_module;

    public:
    MlipModel() = default;
    MlipModel(const std::string& path, torch::Device device);

    auto cast_float_tensors(torch::ScalarType dtype) -> void;
    auto use_only_ensemble_member(std::int64_t member_idx) -> void;
    auto read_extra_features_from_file(const std::string& fpath) -> void;
    auto has_method(const std::string& name) const -> bool;
    auto call_method(const std::string& name, std::vector<torch::jit::IValue> inputs) const
        -> torch::jit::IValue;
    auto forward(std::vector<torch::jit::IValue> inputs) -> torch::jit::IValue;
};
