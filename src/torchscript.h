#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <torch/all.h>
#include <torch/script.h>

#include "mlip.h"

namespace torchscript {

void disableTorchscriptExecutorOptimizations();

/**
 * MLIP implemented as a TorchScript (JIT-compiled) module
 * */
class TorchScriptMlipModel : public mlip::MlipModel {
    torch::jit::Module m_module;
    // Old models take 'charge: int', new ones 'charge: Optional[Tensor]' of shape (N,)
    bool m_legacy_charge_type;

    auto castFloatTensors(torch::ScalarType dtype) -> void;
    auto computeFromMlmmMethod(
        const std::string& method_name,
        std::vector<torch::jit::IValue> inputs,
        const torch::Tensor& ml_coords,
        const torch::Tensor& mm_coords,
        int spin_mult,
        double mm_cutoff,
        const torch::Tensor& pairs_12,
        const torch::Tensor& pairs_13,
        const torch::Tensor& pairs_14,
        bool atomic,
        bool ensemble_values,
        bool calc_forces,
        bool calc_charges,
        bool calc_qbc_and_qbc_grad
    ) -> mlip::MlipOutput;

    public:
    TorchScriptMlipModel(
        const std::string& path, torch::Device device, torch::ScalarType dtype
    );

    auto useOnlyEnsembleMember(std::int64_t member_idx) -> void override;
    auto setExtraFeatures(torch::Tensor extra_features) -> void override;
    auto setStrategy(const std::string& strat) -> void override;
    auto supportsCustomMlmm() const -> bool override;
    auto supportsMultipoleMlmm() const -> bool override;
    auto supportsComputeFromExternalNeighbors() const -> bool override;
    auto computeSinglePoint(
        const torch::Tensor& atomic_numbers,
        const torch::Tensor& coords,
        const std::optional<torch::Tensor>& cell,
        int net_charge,
        int spin_mult,
        bool atomic,
        bool ensemble_values,
        const std::optional<torch::Tensor>& molecule_idxs,
        bool calc_forces,
        bool calc_charges,
        bool calc_qbc_and_qbc_grad
    ) -> mlip::MlipOutput override;
    auto computeFromExternalNeighbors(
        const torch::Tensor& atomic_numbers,
        const torch::Tensor& coords,
        const torch::Tensor& neighborlist,
        const torch::Tensor& shifts,
        int net_charge,
        int spin_mult,
        bool atomic,
        bool ensemble_values,
        const std::optional<torch::Tensor>& molecule_idxs,
        bool calc_forces,
        bool calc_charges,
        bool calc_qbc_and_qbc_grad
    ) -> mlip::MlipOutput override;
    auto computeMlmm(
        const torch::Tensor& atomic_numbers,
        const torch::Tensor& ml_coords,
        const torch::Tensor& mm_coords,
        const torch::Tensor& mm_charges,
        const torch::Tensor& pairs_12,
        const torch::Tensor& pairs_13,
        const torch::Tensor& pairs_14,
        const std::optional<torch::Tensor>& cell,
        int net_charge,
        int spin_mult,
        double mm_cutoff,
        bool atomic,
        bool ensemble_values,
        const std::optional<torch::Tensor>& molecule_idxs,
        bool calc_forces,
        bool calc_charges,
        bool calc_qbc_and_qbc_grad
    ) -> mlip::MlipOutput override;
    auto computeMultipoleMlmm(
        const torch::Tensor& atomic_numbers,
        const torch::Tensor& ml_coords,
        const torch::Tensor& mm_coords,
        const torch::Tensor& mm_charges,
        const torch::Tensor& mm_dipoles,
        const torch::Tensor& mm_quads,
        const torch::Tensor& pairs_12,
        const torch::Tensor& pairs_13,
        const torch::Tensor& pairs_14,
        const std::optional<torch::Tensor>& cell,
        int net_charge,
        int spin_mult,
        double mm_cutoff,
        bool atomic,
        bool ensemble_values,
        const std::optional<torch::Tensor>& molecule_idxs,
        bool calc_forces,
        bool calc_charges,
        bool calc_qbc_and_qbc_grad
    ) -> mlip::MlipOutput override;
};

}  // namespace torchscript
