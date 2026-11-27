#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <torch/all.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include "mlip.h"

namespace compile {

/**
 * MLIP implemented as an AOTInductor-compiled (.pt2) package
 *
 * The package holds one model per device and precision for each computation:
 *
 * "single_point/<device>/<dtype>": (species, coords, net_charge, spin_mult)
 *     -> (energies, forces, atomic_charges)
 * "single_point_pbc/<device>/<dtype>":
 *     (species, coords, cell, pbc, net_charge, spin_mult)
 *     -> (energies, forces, atomic_charges)
 * "external_neighbors/<device>/<dtype>":
 *     (species, coords, neighbor_idxs, shifts, net_charge, spin_mult)
 *     -> (energies, forces, atomic_charges)
 * "custom_mlmm/<device>/<dtype>":
 *     (species, ml_coords, mm_coords, mm_charges, net_charge, spin_mult, mm_cutoff,
 *     pairs_12, pairs_13, pairs_14) -> (energies, forces, mm_forces, atomic_charges)
 * "multipole_mlmm/<device>/<dtype>":
 *     (species, ml_coords, mm_coords, mm_charges, mm_dipoles, mm_quads, net_charge,
 *     spin_mult, mm_cutoff, pairs_12, pairs_13, pairs_14)
 *     -> (energies, forces, mm_forces, atomic_charges)
 *
 * where <device> is "cpu" or "cuda" and <dtype> is "f32" or "f64", net_charge
 * and spin_mult are int64 tensors of shape (molecules,), mm_cutoff is a float tensor
 * of shape (molecules,), and pairs_12, pairs_13, pairs_14 are int64 tensors of shape
 * (2, pairs). Models are loaded on first use.
 * */
class CompiledMlipModel : public mlip::MlipModel {
    using Loader = torch::inductor::AOTIModelPackageLoader;

    std::string m_path;
    torch::Device m_device;
    torch::ScalarType m_dtype;
    bool m_supports_external_neighbors;
    bool m_supports_custom_mlmm;
    bool m_supports_multipole_mlmm;
    std::unique_ptr<Loader> m_single_point_loader;
    std::unique_ptr<Loader> m_pbc_single_point_loader;
    std::unique_ptr<Loader> m_external_neighbors_loader;
    std::unique_ptr<Loader> m_custom_mlmm_loader;
    std::unique_ptr<Loader> m_multipole_mlmm_loader;

    auto modelName(const std::string& prefix) const -> std::string;
    auto makeLoader(const std::string& prefix) const -> std::unique_ptr<Loader>;
    auto perMoleculeTensor(int value, std::int64_t num_molecules) const
        -> torch::Tensor;
    auto runMlmmLoader(
        std::unique_ptr<Loader>& loader,
        const std::string& prefix,
        bool package_has_model,
        const std::vector<torch::Tensor>& inputs,
        const std::optional<torch::Tensor>& cell,
        bool atomic,
        bool ensemble_values,
        const std::optional<torch::Tensor>& molecule_idxs,
        bool calc_forces,
        bool calc_qbc_and_qbc_grad
    ) -> mlip::MlipOutput;

    public:
    CompiledMlipModel(
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
        bool calc_qbc_and_qbc_grad
    ) -> mlip::MlipOutput override;
};

}  // namespace compile
