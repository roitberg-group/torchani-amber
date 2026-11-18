#pragma once

#include <cstdint>
#include <optional>
#include <string>

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

    public:
    TorchScriptMlipModel(
        const std::string& path, torch::Device device, torch::ScalarType dtype
    );

    auto useOnlyEnsembleMember(std::int64_t member_idx) -> void override;
    auto readExtraFeaturesFromFile(const std::string& fpath) -> void override;
    auto setStrategy(const std::string& strat) -> void override;
    auto supportsCustomMlmm() const -> bool override;
    auto supportsComputeFromExternalNeighbors() const -> bool override;
    auto computeSinglePoint(
        const torch::Tensor& atomic_numbers,
        const torch::Tensor& coords,
        const std::optional<torch::Tensor>& cell,
        int net_charge,
        bool atomic,
        bool ensemble_values,
        const std::optional<torch::Tensor>& molecule_idxs,
        bool calc_forces,
        bool calc_charges,
        bool calc_charges_grad,
        bool calc_qbc_and_qbc_grad
    ) -> mlip::MlipOutput override;
    auto computeFromExternalNeighbors(
        const torch::Tensor& atomic_numbers,
        const torch::Tensor& coords,
        const torch::Tensor& neighborlist,
        const torch::Tensor& shifts,
        int net_charge,
        bool atomic,
        bool ensemble_values,
        const std::optional<torch::Tensor>& molecule_idxs,
        bool calc_forces,
        bool calc_charges,
        bool calc_charges_grad,
        bool calc_qbc_and_qbc_grad
    ) -> mlip::MlipOutput override;
    auto computeMlmm(
        const torch::Tensor& atomic_numbers,
        const torch::Tensor& ml_coords,
        const torch::Tensor& mm_coords,
        const torch::Tensor& mm_charges,
        const std::optional<torch::Tensor>& cell,
        int net_charge,
        bool atomic,
        bool ensemble_values,
        const std::optional<torch::Tensor>& molecule_idxs,
        bool calc_forces,
        bool calc_charges,
        bool calc_charges_grad,
        bool calc_qbc_and_qbc_grad
    ) -> mlip::MlipOutput override;
};

}  // namespace torchscript
