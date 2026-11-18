#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/all.h>
#include <torch/script.h>

namespace mlip {

// Keys: "energy", "forces", "atomic_charges", "atomic_charge_grads", "qbc",
// "qbc_grads", "mm_forces". All values are in Hartree and Angstrom.
using MlipOutput = std::unordered_map<std::string, torch::Tensor>;

/**
 * Wrapper around a machine learning interatomic potential
 * */
class MlipModel {
    torch::jit::Module m_module;

    auto castFloatTensors(torch::ScalarType dtype) -> void;
    auto forward(std::vector<torch::jit::IValue> inputs) -> torch::jit::IValue;

    public:
    MlipModel() = default;
    MlipModel(const std::string& path, torch::Device device, torch::ScalarType dtype);

    auto useOnlyEnsembleMember(std::int64_t member_idx) -> void;
    auto readExtraFeaturesFromFile(const std::string& fpath) -> void;
    auto hasMethod(const std::string& name) const -> bool;
    auto supportsCustomMlmm() const -> bool;
    auto supportsComputeFromExternalNeighbors() const -> bool;
    auto callMethod(const std::string& name, std::vector<torch::jit::IValue> inputs) const
        -> torch::jit::IValue;
    auto computeSinglePoint(
        const torch::Tensor& atomic_numbers,
        const torch::Tensor& coords,
        const std::optional<torch::Tensor>& cell = std::nullopt,
        int net_charge = 0,
        bool atomic = false,
        bool ensemble_values = false,
        const std::optional<torch::Tensor>& molecule_idxs = std::nullopt,
        bool calc_forces = false,
        bool calc_charges = false,
        bool calc_charges_grad = false,
        bool calc_qbc_and_qbc_grad = false
    ) -> MlipOutput;
    auto computeFromExternalNeighbors(
        const torch::Tensor& atomic_numbers,
        const torch::Tensor& coords,
        const torch::Tensor& neighborlist,
        const torch::Tensor& shifts,
        bool ensemble_values = false,
        int net_charge = 0,
        const std::optional<torch::Tensor>& molecule_idxs = std::nullopt,
        bool calc_forces = false
    ) -> MlipOutput;
    auto computeMlmm(
        const torch::Tensor& atomic_numbers,
        const torch::Tensor& ml_coords,
        const torch::Tensor& mm_coords,
        const torch::Tensor& mm_charges,
        bool ensemble_values = false,
        int net_charge = 0,
        const std::optional<torch::Tensor>& cell = std::nullopt,
        const std::optional<torch::Tensor>& molecule_idxs = std::nullopt,
        bool calc_forces = false
    ) -> MlipOutput;
};

}  // namespace mlip
