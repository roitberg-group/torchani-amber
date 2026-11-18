#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include <torch/all.h>

namespace mlip {

// Keys: "energy", "forces", "atomic_charges", "atomic_charge_grads", "qbc",
// "qbc_grads", "mm_forces". All values are in Hartree and Angstrom.
using MlipOutput = std::unordered_map<std::string, torch::Tensor>;

enum class ModelFormat { TORCHSCRIPT, PT2_AOTI, PYTHON, AUTO };

/**
 * Interface for a machine learning interatomic potential
 * */
class MlipModel {
    public:
    static auto from_jit_model_path(
        const std::string& path, torch::Device device, torch::ScalarType dtype
    ) -> std::unique_ptr<MlipModel>;
    static auto from_pt2_aoti_model_path(
        const std::string& path, torch::Device device, torch::ScalarType dtype
    ) -> std::unique_ptr<MlipModel>;
    static auto from_python_model_path(
        const std::string& path, torch::Device device, torch::ScalarType dtype
    ) -> std::unique_ptr<MlipModel>;
    static auto from_model_path(
        const std::string& path,
        torch::Device device,
        torch::ScalarType dtype,
        ModelFormat format = ModelFormat::AUTO
    ) -> std::unique_ptr<MlipModel>;

    virtual ~MlipModel() = default;

    virtual auto useOnlyEnsembleMember(std::int64_t member_idx) -> void = 0;
    virtual auto readExtraFeaturesFromFile(const std::string& fpath) -> void = 0;
    virtual auto setStrategy(const std::string& strat) -> void = 0;
    virtual auto supportsCustomMlmm() const -> bool = 0;
    virtual auto supportsComputeFromExternalNeighbors() const -> bool = 0;
    virtual auto computeSinglePoint(
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
    ) -> MlipOutput = 0;
    virtual auto computeFromExternalNeighbors(
        const torch::Tensor& atomic_numbers,
        const torch::Tensor& coords,
        const torch::Tensor& neighborlist,
        const torch::Tensor& shifts,
        int net_charge = 0,
        bool atomic = false,
        bool ensemble_values = false,
        const std::optional<torch::Tensor>& molecule_idxs = std::nullopt,
        bool calc_forces = false,
        bool calc_charges = false,
        bool calc_charges_grad = false,
        bool calc_qbc_and_qbc_grad = false
    ) -> MlipOutput = 0;
    virtual auto computeMlmm(
        const torch::Tensor& atomic_numbers,
        const torch::Tensor& ml_coords,
        const torch::Tensor& mm_coords,
        const torch::Tensor& mm_charges,
        const std::optional<torch::Tensor>& cell = std::nullopt,
        int net_charge = 0,
        bool atomic = false,
        bool ensemble_values = false,
        const std::optional<torch::Tensor>& molecule_idxs = std::nullopt,
        bool calc_forces = false,
        bool calc_charges = false,
        bool calc_charges_grad = false,
        bool calc_qbc_and_qbc_grad = false
    ) -> MlipOutput = 0;
};

}  // namespace mlip
