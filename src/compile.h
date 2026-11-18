#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <torch/all.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include "mlip.h"

namespace compile {

/**
 * MLIP implemented as an AOTInductor-compiled (.pt2) package
 *
 * The package holds one model per device and precision for each computation:
 *
 * "single_point/<device>/<dtype>": (species, coords, net_charge)
 *     -> (energies, forces, atomic_charges)
 * "single_point_pbc/<device>/<dtype>": (species, coords, cell, pbc, net_charge)
 *     -> (energies, forces, atomic_charges)
 * "external_neighbors/<device>/<dtype>":
 *     (species, coords, neighbor_idxs, shifts, net_charge)
 *     -> (energies, forces, atomic_charges)
 * "custom_mlmm/<device>/<dtype>": (species, ml_coords, mm_coords, mm_charges, net_charge)
 *     -> (energies, forces, mm_forces, atomic_charges)
 *
 * where <device> is "cpu" or "cuda" and <dtype> is "f32" or "f64", and net_charge
 * is an int64 tensor of shape (molecules,). Models are loaded on first use.
 * */
class CompiledMlipModel : public mlip::MlipModel {
    using Loader = torch::inductor::AOTIModelPackageLoader;

    std::string m_path;
    torch::Device m_device;
    torch::ScalarType m_dtype;
    bool m_supports_external_neighbors;
    bool m_supports_custom_mlmm;
    std::unique_ptr<Loader> m_single_point_loader;
    std::unique_ptr<Loader> m_pbc_single_point_loader;
    std::unique_ptr<Loader> m_external_neighbors_loader;
    std::unique_ptr<Loader> m_custom_mlmm_loader;

    auto modelName(const std::string& prefix) const -> std::string;
    auto makeLoader(const std::string& prefix) const -> std::unique_ptr<Loader>;
    auto netChargeTensor(int net_charge, std::int64_t num_molecules) const -> torch::Tensor;

    public:
    CompiledMlipModel(
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

}  // namespace compile
