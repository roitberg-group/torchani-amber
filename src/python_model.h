#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include <pybind11/pybind11.h>
#include <torch/all.h>

#include "mlip_model.h"

namespace python_model {

/**
 * MLIP implemented as a Python module, run by an embedded Python interpreter
 *
 * The model is a directory with a model.py that defines
 * make_model(device: str, dtype: torch.dtype) -> torch.nn.Module. The module has a
 * 'forward', and optionally 'compute_from_external_neighbors', 'compute_mlmm',
 * 'compute_multipole_mlmm', 'set_active_members', 'set_extra_scalar_atomic_features' and
 * 'set_strategy'. The leading arguments (species_coordinates, or species, coords,
 * neighbor_idxs, shifts, or the ML/MM coords and charges) are positional, the rest are
 * passed by name. 'charge' is an int if annotated as such in 'forward', otherwise it
 * and 'spin_mult' are int64 tensors with one value per molecule. All methods return a
 * dict with "energies" and extra outputs, the energies, or a tuple (energies, extra) or
 * (species, energies, extra), where extra is optional, and None, the atomic charges, or
 * a dict of tensors.
 *
 * If model.py imports torchani, the torchani package loads its own cuaev.so and
 * cell_list.so. If those are different files from the ones libtorchani already
 * loaded (e.g. libtorchani run from a build tree, with another copy installed), the
 * ops are registered twice and the process aborts ("Only a single TORCH_LIBRARY can
 * be used to register the namespace cuaev").
 * */
// Hidden like pybind11's own types, since it holds a pybind11::object
class __attribute__((visibility("hidden"))) PythonMlipModel : public mlip::MlipModel {
    pybind11::object m_model;
    // Old models take 'charge: int', new ones 'charge: Tensor | None' of shape (N,)
    bool m_legacy_charge_type = false;

    auto hasMethod(const char* name) const -> bool;
    auto computeMlmmCoreImpl(
        const char* method_name,
        const torch::Tensor& atomic_numbers,
        const torch::Tensor& ml_coords,
        const torch::Tensor& mm_coords,
        const torch::Tensor& mm_charges,
        const std::optional<torch::Tensor>& mm_dipoles,
        const std::optional<torch::Tensor>& mm_quads,
        const std::optional<torch::Tensor>& cell,
        int net_charge,
        int spin_mult,
        double mm_cutoff,
        bool atomic,
        bool ensemble_values,
        const std::optional<torch::Tensor>& molecule_idxs,
        bool calc_forces,
        bool calc_qbc_and_qbc_grad
    ) -> mlip::MlipOutput;

    public:
    PythonMlipModel(
        const std::string& model_dir,
        torch::Device device,
        torch::ScalarType dtype,
        const mlip::ModelOptions& options
    );
    ~PythonMlipModel() override;
    PythonMlipModel(const PythonMlipModel&) = delete;
    auto operator=(const PythonMlipModel&) -> PythonMlipModel& = delete;

    auto useOnlyEnsembleMember(std::int64_t member_idx) -> void override;
    auto setExtraScalarAtomicFeatures(torch::Tensor extra_scalar_atomic_features)
        -> void override;
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

}  // namespace python_model
