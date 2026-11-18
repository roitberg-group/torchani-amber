#include "mlip.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <c10/util/Logging.h>
#include <torch/all.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/script.h>

namespace {
auto setupInputs(
    const torch::Tensor& atomic_numbers,
    const torch::Tensor& coords,
    const std::optional<torch::Tensor>& cell,
    int net_charge,
    bool atomic,
    bool ensemble_values,
    const std::optional<torch::Tensor>& molecule_idxs
) -> std::vector<torch::jit::IValue> {
    std::optional<torch::Tensor> pbc;
    if (cell.has_value()) {
        pbc = torch::tensor(
            {true, true, true},
            torch::TensorOptions().dtype(torch::kBool).device(cell->device())
        );
    }
    return {
        std::tuple{atomic_numbers, coords},
        /* cell= */ cell,
        /* pbc= */ pbc,
        /* charge= */ net_charge,
        /* atomic= */ atomic,
        /* ensemble_values= */ ensemble_values,
        /* molecule_idxs= */ molecule_idxs
    };
}

auto setupInputsExternalNeighbors(
    const torch::Tensor& atomic_numbers,
    const torch::Tensor& coords,
    const torch::Tensor& neighborlist,
    const torch::Tensor& shifts,
    bool ensemble_values,
    int net_charge,
    const std::optional<torch::Tensor>& molecule_idxs
) -> std::vector<torch::jit::IValue> {
    return {
        atomic_numbers,
        coords,
        neighborlist,
        shifts,
        /* charge= */ net_charge,
        /* atomic= */ false,
        /* ensemble_values= */ ensemble_values,
        // An empty torch::jit::IValue() is equivalent to torch::indexing::None
        /* molecule_idxs= */ molecule_idxs
    };
}

auto setupInputsCustomMlmm(
    const torch::Tensor& atomic_numbers,
    const torch::Tensor& ml_coords,
    const torch::Tensor& mm_coords,
    const torch::Tensor& mm_charges,
    bool ensemble_values,
    int net_charge,
    const std::optional<torch::Tensor>& cell,
    const std::optional<torch::Tensor>& molecule_idxs
) -> std::vector<torch::jit::IValue> {
    std::optional<torch::Tensor> pbc;
    if (cell.has_value()) {
        pbc = torch::tensor(
            {true, true, true},
            torch::TensorOptions().dtype(torch::kBool).device(cell->device())
        );
    }
    return {
        std::tuple{atomic_numbers, ml_coords},
        /* mm_coords= */ mm_coords,
        /* mm_charges= */ mm_charges,
        /* cell= */ cell,
        /* pbc= */ pbc,
        /* charge= */ net_charge,
        /* atomic= */ false,
        /* ensemble_values= */ ensemble_values,
        /* molecule_idxs= */ molecule_idxs
    };
}
torch::Tensor fetchEnergiesFromOutput(const torch::jit::IValue& output) {
    // New signature for compute_from_neighbors
    if (!output.isTuple()) {
        return output.toGenericDict().at("energies").toTensor();
    }

    // HACK to support legacy compute_from_neighbors signature (energies, scalars)
    torch::Tensor first_tensor = output.toTuple()->elements()[0].toTensor();
    if (first_tensor.is_floating_point()) {
        return first_tensor;
    }

    // Standard signature
    return output.toTuple()->elements()[1].toTensor();
}

torch::Tensor fetchAtomicChargesFromOutput(const torch::jit::IValue& output) {
    // New signature for compute_from_neighbors
    if (!output.isTuple()) {
        return output.toGenericDict().at("atomic_charges").toTensor();
    }
    // HACK to support legacy compute_from_neighbors signature (energies, scalars)
    torch::Tensor first_tensor = output.toTuple()->elements()[0].toTensor();
    if (first_tensor.is_floating_point()) {
        return output.toTuple()->elements()[1].toTensor();
    }

    // Standard signature, tuple form or tuple-dict form
    const auto& scalars = output.toTuple()->elements()[2];
    if (scalars.isTensor()) {
        return scalars.toTensor();
    }
    return scalars.toGenericDict().at("atomic_charges").toTensor();
}

void validateModelOutput(
    const torch::jit::IValue& output, size_t expect_len, bool external = false
) {
    try {
        output.toTuple();
    } catch (const c10::Error& e) {
        std::cerr << "ERROR (libtorchani)\n"
                  << "Expected model to return a tuple of tensors" << std::endl;
#ifdef DEBUG
        throw;
#endif
        std::exit(2);
    }
    if (output.toTuple()->elements().size() < expect_len) {
        std::cerr << "ERROR (libtorchani)\n";
        std::cerr << "Expected model to return ";
        if (expect_len == 2) {
            if (external) {
                std::cerr << "a tuple (energies, atomic_scalars)";
            } else {
                std::cerr << "a tuple (species, energies, ...)";
            }
        } else if (expect_len == 3) {
            std::cerr << "a tuple (species, energies, atomic_charges, ...) or (species, energies, scalars)";
        } else {
            std::cerr << "a tuple of len at least " << expect_len;
        }
        std::cerr << std::endl;
        std::exit(2);
    }
}
}  // namespace

namespace mlip {

MlipModel::MlipModel(
    const std::string& path, torch::Device device, torch::ScalarType dtype
)
    : m_module{torch::jit::load(path, device)} {
    castFloatTensors(dtype);
}

// NOTE: Unfortunately, due to a strange LibTorch design choice, Module::to() converts
// *all params and bufs* to the dtype, not only the float tensors like the Python API.
// Here we loop over the params and buffers manually and convert only the floating
// point ones
auto MlipModel::castFloatTensors(torch::ScalarType dtype) -> void {
    for (at::Tensor t : m_module.parameters()) {
        if (t.is_floating_point()) {
            t.set_data(t.to(dtype));
        }
    }
    for (at::Tensor t : m_module.buffers()) {
        if (t.is_floating_point()) {
            t.set_data(t.to(dtype));
        }
    }
    LOG(INFO) << "Cast MLIP model to the specified precision.";
}

auto MlipModel::useOnlyEnsembleMember(std::int64_t member_idx) -> void {
    if (not m_module.find_method("set_active_members").has_value()) {
        throw std::runtime_error(
            "Model doesn't export a method 'set_active_members(list[int]) -> None'"
        );
    }
    m_module.get_method("set_active_members")({torch::List<std::int64_t>{member_idx}});
}

auto MlipModel::readExtraFeaturesFromFile(const std::string& fpath) -> void {
    if (not m_module.find_method("set_extra_features").has_value()) {
        throw std::runtime_error(
            "Model doesn't export a method 'set_extra_features(list[float]) -> None'"
        );
    }
    // TODO: This is ugly, it would be better to fail gracefully if the file can't
    // be parsed
    std::ifstream features_file(fpath);
    torch::List<double> features;
    double val;
    while (features_file >> val) {
        features.push_back(val);
    }
    m_module.get_method("set_extra_features")({features});
}

auto MlipModel::supportsCustomMlmm() const -> bool {
    return m_module.find_method("compute_mlmm").has_value();
}

auto MlipModel::supportsComputeFromExternalNeighbors() const -> bool {
    return m_module.find_method("compute_from_external_neighbors").has_value();
}

auto MlipModel::setStrategy(const std::string& strat) -> void {
    // If the model doesn't export set_strategy, then this is just a no-op. It is not
    // required that models export set_strategy
    if (not m_module.find_method("set_strategy").has_value()) {
        return;
    }
    try {
        m_module.get_method("set_strategy")({strat});
    } catch (const c10::Error& e) {
        throw std::runtime_error(
            "The selected model exports 'set_strategy(str) -> None'"
            " but it doesn't accept the specified strategy"
        );
    }
}

auto MlipModel::computeSinglePoint(
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
) -> MlipOutput {
    if (atomic) {
        throw std::runtime_error("computeSinglePoint doesn't support atomic=true");
    }
    if (ensemble_values) {
        throw std::runtime_error("computeSinglePoint doesn't support ensemble_values=true");
    }
    if (calc_charges_grad and not calc_charges) {
        throw std::runtime_error("calc_charges_grad=true requires calc_charges=true");
    }
    torch::jit::IValue raw_output = m_module.forward(setupInputs(
        atomic_numbers,
        coords,
        cell,
        net_charge,
        /* atomic= */ false,
        /* ensemble_values= */ calc_qbc_and_qbc_grad,
        molecule_idxs
    ));
    validateModelOutput(raw_output, calc_charges ? 3 : 2);

    MlipOutput output;
    torch::Tensor energy = fetchEnergiesFromOutput(raw_output);
    if (calc_qbc_and_qbc_grad) {
        torch::Tensor num_atoms = torch::tensor(
            coords.size(1),
            torch::TensorOptions().dtype(coords.dtype()).device(coords.device())
        );
        output["qbc"] = torch::std(energy, 0) / num_atoms.sqrt();
        energy = energy.mean(0);
    }
    output["energy"] = energy;

    if (calc_charges) {
        torch::Tensor atomic_charges = fetchAtomicChargesFromOutput(raw_output);
        // Old TorchANI outputs charges as (1, molecs, atoms), new as (molecs, atoms)
        if (atomic_charges.dim() == 3) {
            atomic_charges = atomic_charges.squeeze(0);
        }
        output["atomic_charges"] = atomic_charges;
    }

    if (calc_forces) {
        output["forces"] = -torch::autograd::grad(
            {energy.sum()},
            {coords},
            /* grad_outputs= */ {},
            /* retain_graph= */ calc_qbc_and_qbc_grad or calc_charges_grad
        )[0];
    }
    if (calc_qbc_and_qbc_grad) {
        output["qbc_grads"] = torch::autograd::grad(
            {output.at("qbc").sum()},
            {coords},
            /* grad_outputs= */ {},
            /* retain_graph= */ calc_charges_grad
        )[0];
    }
    if (calc_charges_grad) {
        torch::Tensor atomic_charges = output.at("atomic_charges");
        std::vector<torch::Tensor> charge_grads;
        for (std::int64_t atom_idx = 0; atom_idx != coords.size(1); ++atom_idx) {
            torch::Tensor charge_grad = torch::autograd::grad(
                {atomic_charges.index({0, atom_idx})},
                {coords},
                /* grad_outputs= */ {},
                /* retain_graph= */ true
            )[0];
            charge_grads.push_back(charge_grad.squeeze(0));
        }
        output["atomic_charge_grads"] = torch::stack(charge_grads);
    }
    return output;
}

auto MlipModel::computeFromExternalNeighbors(
    const torch::Tensor& atomic_numbers,
    const torch::Tensor& coords,
    const torch::Tensor& neighborlist,
    const torch::Tensor& shifts,
    bool ensemble_values,
    int net_charge,
    const std::optional<torch::Tensor>& molecule_idxs,
    bool calc_forces
) -> MlipOutput {
    if (not supportsComputeFromExternalNeighbors()) {
        throw std::runtime_error(
            "Model doesn't export a method 'compute_from_external_neighbors(...)'"
        );
    }
    if (ensemble_values) {
        throw std::runtime_error(
            "computeFromExternalNeighbors doesn't support ensemble_values=true"
        );
    }
    torch::jit::IValue raw_output = m_module.get_method("compute_from_external_neighbors")(
        setupInputsExternalNeighbors(
            atomic_numbers,
            coords,
            neighborlist,
            shifts,
            ensemble_values,
            net_charge,
            molecule_idxs
        )
    );
    MlipOutput output;
    torch::Tensor energy = fetchEnergiesFromOutput(raw_output);
    output["energy"] = energy;
    if (calc_forces) {
        output["forces"] = -torch::autograd::grad(
            {energy.sum()},
            {coords},
            /* grad_outputs= */ {},
            /* retain_graph= */ false
        )[0];
    }
    return output;
}

auto MlipModel::computeMlmm(
    const torch::Tensor& atomic_numbers,
    const torch::Tensor& ml_coords,
    const torch::Tensor& mm_coords,
    const torch::Tensor& mm_charges,
    bool ensemble_values,
    int net_charge,
    const std::optional<torch::Tensor>& cell,
    const std::optional<torch::Tensor>& molecule_idxs,
    bool calc_forces
) -> MlipOutput {
    if (not supportsCustomMlmm()) {
        throw std::runtime_error("Model doesn't export a method 'compute_mlmm(...)'");
    }
    if (ensemble_values) {
        throw std::runtime_error("computeMlmm doesn't support ensemble_values=true");
    }
    torch::jit::IValue raw_output = m_module.get_method("compute_mlmm")(
        setupInputsCustomMlmm(
            atomic_numbers,
            ml_coords,
            mm_coords,
            mm_charges,
            ensemble_values,
            net_charge,
            cell,
            molecule_idxs
        )
    );
    MlipOutput output;
    torch::Tensor energy = raw_output.toTensor();
    output["energy"] = energy;
    if (calc_forces) {
        std::vector<torch::Tensor> grads = torch::autograd::grad(
            {energy.sum()},
            {ml_coords, mm_coords},
            /* grad_outputs= */ {},
            /* retain_graph= */ false
        );
        output["forces"] = -grads[0];
        output["mm_forces"] = -grads[1];
    }
    return output;
}

}  // namespace mlip
