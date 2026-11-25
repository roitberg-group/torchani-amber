#include "torchscript.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <c10/util/Logging.h>
#include <torch/all.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/script.h>

#include "mlip.h"

namespace {
auto chargeArgIsInt(const torch::jit::Module& module) -> bool {
    const c10::FunctionSchema& schema = module.get_method("forward").function().getSchema();
    std::optional<int> charge_idx = schema.argumentIndexWithName("charge");
    if (not charge_idx.has_value()) {
        throw std::runtime_error("Model's 'forward' has no 'charge' argument");
    }
    return schema.arguments().at(*charge_idx).type()->kind() == c10::TypeKind::IntType;
}

auto chargeInput(int net_charge, const torch::Tensor& coords, bool legacy_charge_type)
    -> torch::jit::IValue {
    if (legacy_charge_type) {
        return net_charge;
    }
    return torch::full({coords.size(0)}, net_charge, coords.options().dtype(torch::kLong));
}

// Models whose method has no 'spin_mult' argument only support spin_mult = 1
auto insertSpinMultInput(
    const torch::jit::Method& method,
    std::vector<torch::jit::IValue>& inputs,
    int spin_mult,
    const torch::Tensor& coords
) -> void {
    std::optional<int> idx = method.function().getSchema().argumentIndexWithName("spin_mult");
    if (idx.has_value()) {
        // The schema's first argument is 'self', which is not part of the inputs
        inputs.insert(
            inputs.begin() + *idx - 1,
            torch::full({coords.size(0)}, spin_mult, coords.options().dtype(torch::kLong))
        );
    } else if (spin_mult != 1) {
        throw std::runtime_error(
            "The model's '" + method.name()
            + "' has no 'spin_mult' argument, only spin_mult = 1 is supported"
        );
    }
}

auto setupInputs(
    const torch::Tensor& atomic_numbers,
    const torch::Tensor& coords,
    const std::optional<torch::Tensor>& cell,
    int net_charge,
    bool legacy_charge_type,
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
        /* charge= */ chargeInput(net_charge, coords, legacy_charge_type),
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
    bool legacy_charge_type,
    const std::optional<torch::Tensor>& molecule_idxs
) -> std::vector<torch::jit::IValue> {
    return {
        atomic_numbers,
        coords,
        neighborlist,
        shifts,
        /* charge= */ chargeInput(net_charge, coords, legacy_charge_type),
        /* atomic= */ false,
        /* ensemble_values= */ ensemble_values,
        // An empty torch::jit::IValue() is equivalent to torch::indexing::None
        /* molecule_idxs= */ molecule_idxs
    };
}

// mm_dipoles and mm_quads are only passed to 'compute_multipole_mlmm'
auto setupInputsCustomMlmm(
    const torch::Tensor& atomic_numbers,
    const torch::Tensor& ml_coords,
    const torch::Tensor& mm_coords,
    const torch::Tensor& mm_charges,
    const std::optional<torch::Tensor>& mm_dipoles,
    const std::optional<torch::Tensor>& mm_quads,
    bool ensemble_values,
    int net_charge,
    bool legacy_charge_type,
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
    std::vector<torch::jit::IValue> inputs = {
        std::tuple{atomic_numbers, ml_coords},
        /* mm_coords= */ mm_coords,
        /* mm_charges= */ mm_charges,
    };
    if (mm_dipoles.has_value()) {
        inputs.push_back(*mm_dipoles);
    }
    if (mm_quads.has_value()) {
        inputs.push_back(*mm_quads);
    }
    inputs.push_back(/* cell= */ cell);
    inputs.push_back(/* pbc= */ pbc);
    inputs.push_back(/* charge= */ chargeInput(net_charge, ml_coords, legacy_charge_type));
    inputs.push_back(/* atomic= */ false);
    inputs.push_back(/* ensemble_values= */ ensemble_values);
    inputs.push_back(/* molecule_idxs= */ molecule_idxs);
    return inputs;
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

// Outputs of the model other than the species and the energies, e.g. "atomic_charges"
auto fetchExtraOutputs(const torch::jit::IValue& output) -> mlip::MlipOutput {
    mlip::MlipOutput extra_outputs;
    // New signature for compute_from_neighbors
    if (!output.isTuple()) {
        for (const auto& item : output.toGenericDict()) {
            const std::string& key = item.key().toStringRef();
            if (key != "energies" && item.value().isTensor()) {
                extra_outputs[key] = item.value().toTensor();
            }
        }
        return extra_outputs;
    }
    const auto& elements = output.toTuple()->elements();
    // HACK to support legacy compute_from_neighbors signature (energies, scalars)
    std::size_t extra_idx = 2;
    if (elements[0].toTensor().is_floating_point()) {
        extra_idx = 1;
    }
    // Standard signature, tuple form or tuple-dict form
    if (elements.size() <= extra_idx || elements[extra_idx].isNone()) {
        return extra_outputs;
    }
    const auto& extra = elements[extra_idx];
    if (extra.isTensor()) {
        extra_outputs["atomic_charges"] = extra.toTensor();
        return extra_outputs;
    }
    for (const auto& item : extra.toGenericDict()) {
        extra_outputs[item.key().toStringRef()] = item.value().toTensor();
    }
    return extra_outputs;
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

// Energies, extra outputs of the model, and optionally forces, QBC and QBC grads wrt
// the coords. If mm_coords is passed, the forces on the MM atoms are also calculated
auto computeOutput(
    torch::Tensor energy,
    mlip::MlipOutput extra_outputs,
    const torch::Tensor& coords,
    const std::optional<torch::Tensor>& mm_coords,
    bool calc_forces,
    bool calc_qbc_and_qbc_grad
) -> mlip::MlipOutput {
    mlip::MlipOutput output = std::move(extra_outputs);
    if (calc_qbc_and_qbc_grad) {
        torch::Tensor num_atoms = torch::tensor(
            coords.size(1),
            torch::TensorOptions().dtype(coords.dtype()).device(coords.device())
        );
        output["qbc"] = torch::std(energy, 0) / num_atoms.sqrt();
        energy = energy.mean(0);
    }
    output["energy"] = energy;

    // Old TorchANI outputs charges as (1, molecs, atoms), new as (molecs, atoms)
    if (output.count("atomic_charges") > 0 && output.at("atomic_charges").dim() == 3) {
        output["atomic_charges"] = output.at("atomic_charges").squeeze(0);
    }

    if (calc_forces) {
        bool retain_graph = calc_qbc_and_qbc_grad;
        if (mm_coords.has_value()) {
            std::vector<torch::Tensor> grads = torch::autograd::grad(
                {energy.sum()},
                {coords, *mm_coords},
                /* grad_outputs= */ {},
                retain_graph
            );
            output["forces"] = -grads[0];
            output["mm_forces"] = -grads[1];
        } else {
            output["forces"] = -torch::autograd::grad(
                {energy.sum()}, {coords}, /* grad_outputs= */ {}, retain_graph
            )[0];
        }
    }
    if (calc_qbc_and_qbc_grad) {
        output["qbc_grads"] = torch::autograd::grad(
            {output.at("qbc").sum()},
            {coords},
            /* grad_outputs= */ {},
            // Without calc_forces, the caller may differentiate the energy afterwards
            /* retain_graph= */ not calc_forces
        )[0];
    }
    return output;
}
}  // namespace

namespace torchscript {

void disableTorchscriptExecutorOptimizations() {
    torch::jit::setGraphExecutorOptimize(false);
    LOG(INFO) << "Disabled TorchScript graph executor optimizations";
}

TorchScriptMlipModel::TorchScriptMlipModel(
    const std::string& path, torch::Device device, torch::ScalarType dtype
)
    : m_module{torch::jit::load(path, device)},
      m_legacy_charge_type{chargeArgIsInt(m_module)} {
    castFloatTensors(dtype);
}

// NOTE: Unfortunately, due to a strange LibTorch design choice, Module::to() converts
// *all params and bufs* to the dtype, not only the float tensors like the Python API.
// Here we loop over the params and buffers manually and convert only the floating
// point ones
auto TorchScriptMlipModel::castFloatTensors(torch::ScalarType dtype) -> void {
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

auto TorchScriptMlipModel::useOnlyEnsembleMember(std::int64_t member_idx) -> void {
    if (not m_module.find_method("set_active_members").has_value()) {
        throw std::runtime_error(
            "Model doesn't export a method 'set_active_members(list[int]) -> None'"
        );
    }
    m_module.get_method("set_active_members")({torch::List<std::int64_t>{member_idx}});
}

auto TorchScriptMlipModel::setExtraFeatures(torch::Tensor extra_features) -> void {
    if (not m_module.find_method("set_extra_features").has_value()) {
        throw std::runtime_error(
            "Model doesn't export a method 'set_extra_features(list[float]) -> None'"
        );
    }
    extra_features = extra_features.to(torch::kCPU, torch::kDouble).contiguous();
    auto extra_features_a = extra_features.accessor<double, 1>();
    torch::List<double> features;
    for (std::int64_t i = 0; i != extra_features.size(0); ++i) {
        features.push_back(extra_features_a[i]);
    }
    m_module.get_method("set_extra_features")({features});
}

auto TorchScriptMlipModel::supportsCustomMlmm() const -> bool {
    return m_module.find_method("compute_mlmm").has_value();
}

auto TorchScriptMlipModel::supportsMultipoleMlmm() const -> bool {
    return m_module.find_method("compute_multipole_mlmm").has_value();
}

auto TorchScriptMlipModel::supportsComputeFromExternalNeighbors() const -> bool {
    return m_module.find_method("compute_from_external_neighbors").has_value();
}

auto TorchScriptMlipModel::setStrategy(const std::string& strat) -> void {
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

auto TorchScriptMlipModel::computeSinglePoint(
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
) -> mlip::MlipOutput {
    if (atomic) {
        throw std::runtime_error("computeSinglePoint doesn't support atomic=true");
    }
    if (ensemble_values) {
        throw std::runtime_error("computeSinglePoint doesn't support ensemble_values=true");
    }
    std::vector<torch::jit::IValue> inputs = setupInputs(
        atomic_numbers,
        coords,
        cell,
        net_charge,
        m_legacy_charge_type,
        /* atomic= */ false,
        /* ensemble_values= */ calc_qbc_and_qbc_grad,
        molecule_idxs
    );
    torch::jit::Method method = m_module.get_method("forward");
    insertSpinMultInput(method, inputs, spin_mult, coords);
    torch::jit::IValue raw_output = method(inputs);
    validateModelOutput(raw_output, 2);
    return computeOutput(
        fetchEnergiesFromOutput(raw_output),
        fetchExtraOutputs(raw_output),
        coords,
        /* mm_coords= */ std::nullopt,
        calc_forces,
        calc_qbc_and_qbc_grad
    );
}

auto TorchScriptMlipModel::computeFromExternalNeighbors(
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
) -> mlip::MlipOutput {
    if (not supportsComputeFromExternalNeighbors()) {
        throw std::runtime_error(
            "Model doesn't export a method 'compute_from_external_neighbors(...)'"
        );
    }
    if (atomic) {
        throw std::runtime_error("computeFromExternalNeighbors doesn't support atomic=true");
    }
    if (ensemble_values) {
        throw std::runtime_error(
            "computeFromExternalNeighbors doesn't support ensemble_values=true"
        );
    }
    std::vector<torch::jit::IValue> inputs = setupInputsExternalNeighbors(
        atomic_numbers,
        coords,
        neighborlist,
        shifts,
        /* ensemble_values= */ calc_qbc_and_qbc_grad,
        net_charge,
        m_legacy_charge_type,
        molecule_idxs
    );
    torch::jit::Method method = m_module.get_method("compute_from_external_neighbors");
    insertSpinMultInput(method, inputs, spin_mult, coords);
    torch::jit::IValue raw_output = method(inputs);
    return computeOutput(
        fetchEnergiesFromOutput(raw_output),
        fetchExtraOutputs(raw_output),
        coords,
        /* mm_coords= */ std::nullopt,
        calc_forces,
        calc_qbc_and_qbc_grad
    );
}

auto TorchScriptMlipModel::computeMlmm(
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
) -> mlip::MlipOutput {
    std::vector<torch::jit::IValue> inputs = setupInputsCustomMlmm(
        atomic_numbers,
        ml_coords,
        mm_coords,
        mm_charges,
        /* mm_dipoles= */ std::nullopt,
        /* mm_quads= */ std::nullopt,
        /* ensemble_values= */ calc_qbc_and_qbc_grad,
        net_charge,
        m_legacy_charge_type,
        cell,
        molecule_idxs
    );
    return computeFromMlmmMethod(
        "compute_mlmm",
        inputs,
        ml_coords,
        mm_coords,
        spin_mult,
        mm_cutoff,
        pairs_12,
        pairs_13,
        pairs_14,
        atomic,
        ensemble_values,
        calc_forces,
        calc_qbc_and_qbc_grad
    );
}

auto TorchScriptMlipModel::computeMultipoleMlmm(
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
) -> mlip::MlipOutput {
    std::vector<torch::jit::IValue> inputs = setupInputsCustomMlmm(
        atomic_numbers,
        ml_coords,
        mm_coords,
        mm_charges,
        mm_dipoles,
        mm_quads,
        /* ensemble_values= */ calc_qbc_and_qbc_grad,
        net_charge,
        m_legacy_charge_type,
        cell,
        molecule_idxs
    );
    return computeFromMlmmMethod(
        "compute_multipole_mlmm",
        inputs,
        ml_coords,
        mm_coords,
        spin_mult,
        mm_cutoff,
        pairs_12,
        pairs_13,
        pairs_14,
        atomic,
        ensemble_values,
        calc_forces,
        calc_qbc_and_qbc_grad
    );
}

auto TorchScriptMlipModel::computeFromMlmmMethod(
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
    bool calc_qbc_and_qbc_grad
) -> mlip::MlipOutput {
    if (not m_module.find_method(method_name).has_value()) {
        throw std::runtime_error("Model doesn't export a method '" + method_name + "(...)'");
    }
    if (atomic) {
        throw std::runtime_error(method_name + " doesn't support atomic=true");
    }
    if (ensemble_values) {
        throw std::runtime_error(method_name + " doesn't support ensemble_values=true");
    }
    torch::jit::Method method = m_module.get_method(method_name);
    torch::TensorOptions long_options = ml_coords.options().dtype(torch::kLong);
    // (name, value, value is the default)
    std::vector<std::tuple<std::string, torch::Tensor, bool>> extra_inputs = {
        {"spin_mult",
         torch::full({ml_coords.size(0)}, spin_mult, long_options),
         spin_mult == 1},
        {"mm_cutoff",
         torch::full({ml_coords.size(0)}, mm_cutoff, ml_coords.options()),
         mm_cutoff == 0.0},
        {"pairs_12", pairs_12, pairs_12.numel() == 0},
        {"pairs_13", pairs_13, pairs_13.numel() == 0},
        {"pairs_14", pairs_14, pairs_14.numel() == 0},
    };
    const c10::FunctionSchema& schema = method.function().getSchema();
    // Sorted by position, so that each insertion leaves the next position valid
    std::map<int, torch::Tensor> positioned_extra_inputs;
    for (const auto& [name, value, is_default] : extra_inputs) {
        std::optional<int> idx = schema.argumentIndexWithName(name);
        if (idx.has_value()) {
            // The schema's first argument is 'self', which is not part of the inputs
            positioned_extra_inputs[*idx - 1] = value;
        } else if (not is_default) {
            throw std::runtime_error(
                "The model's '" + method_name + "' has no '" + name
                + "' argument, only its default value is supported"
            );
        }
    }
    for (const auto& [idx, value] : positioned_extra_inputs) {
        inputs.insert(inputs.begin() + idx, value);
    }
    torch::jit::IValue raw_output = method(inputs);
    // The method returns either the energies, or a tuple (energies, atomic_charges)
    torch::Tensor energy;
    mlip::MlipOutput extra_outputs;
    if (raw_output.isTensor()) {
        energy = raw_output.toTensor();
    } else {
        const auto& elements = raw_output.toTuple()->elements();
        energy = elements.at(0).toTensor();
        if (not elements.at(1).isNone()) {
            extra_outputs["atomic_charges"] = elements.at(1).toTensor();
        }
    }
    return computeOutput(
        energy,
        extra_outputs,
        ml_coords,
        mm_coords,
        calc_forces,
        calc_qbc_and_qbc_grad
    );
}

}  // namespace torchscript
