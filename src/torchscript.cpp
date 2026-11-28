#include "torchscript.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <c10/util/Logging.h>
#include <torch/all.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/script.h>

#include "mlip_model.h"
#include "model_output.h"

using model_output::computeOutput;

namespace {
auto methodChargeArgIsInt(const torch::jit::Method& method) -> bool {
    const c10::FunctionSchema& schema = method.function().getSchema();
    std::optional<int> charge_idx = schema.argumentIndexWithName("charge");
    if (not charge_idx.has_value()) {
        throw std::runtime_error(
            "Model's '" + method.name() + "' has no 'charge' argument"
        );
    }
    return schema.arguments().at(*charge_idx).type()->kind() == c10::TypeKind::IntType;
}

auto chargeArgIsInt(const torch::jit::Module& module) -> bool {
    bool charge_is_int = methodChargeArgIsInt(module.get_method("forward"));
    const std::array<const char*, 3> other_methods = {
        "compute_from_external_neighbors",
        "compute_mlmm",
        "compute_multipole_mlmm",
    };
    for (const char* name : other_methods) {
        std::optional<torch::jit::Method> method = module.find_method(name);
        if (method.has_value() and methodChargeArgIsInt(*method) != charge_is_int) {
            throw std::runtime_error("Inconsistent 'charge' type in different methods");
        }
    }
    return charge_is_int;
}

auto chargeInput(int net_charge, const torch::Tensor& coords, bool legacy_charge_type)
    -> torch::jit::IValue {
    if (legacy_charge_type) {
        return net_charge;
    }
    return torch::full(
        {coords.size(0)}, net_charge, coords.options().dtype(torch::kLong)
    );
}

auto pbcFromCell(const std::optional<torch::Tensor>& cell)
    -> std::optional<torch::Tensor> {
    if (not cell.has_value()) {
        return std::nullopt;
    }
    return torch::tensor(
        {true, true, true},
        torch::TensorOptions().dtype(torch::kBool).device(cell->device())
    );
}

// The arguments after the leading ones are passed by name, as for Python models.
// 'molecule_idxs' is only passed if it has a value, since old models name it
// '_molecule_idxs'
auto commonKwargs(
    const torch::Tensor& coords,
    int net_charge,
    bool legacy_charge_type,
    bool ensemble_values,
    const std::optional<torch::Tensor>& molecule_idxs
) -> torch::jit::Kwargs {
    torch::jit::Kwargs kwargs;
    kwargs["charge"] = chargeInput(net_charge, coords, legacy_charge_type);
    kwargs["atomic"] = false;
    kwargs["ensemble_values"] = ensemble_values;
    if (molecule_idxs.has_value()) {
        kwargs["molecule_idxs"] = *molecule_idxs;
    }
    return kwargs;
}

// Models whose method has no 'spin_mult' argument only support spin_mult = 1
auto insertSpinMultKwarg(
    const torch::jit::Method& method,
    torch::jit::Kwargs& kwargs,
    int spin_mult,
    const torch::Tensor& coords
) -> void {
    if (method.function().getSchema().argumentIndexWithName("spin_mult").has_value()) {
        kwargs["spin_mult"] = torch::full(
            {coords.size(0)}, spin_mult, coords.options().dtype(torch::kLong)
        );
    } else if (spin_mult != 1) {
        throw std::runtime_error(
            "The model's '" + method.name()
            + "' has no 'spin_mult' argument, only spin_mult = 1 is supported"
        );
    }
}

// mm_dipoles and mm_quads are only passed to 'compute_multipole_mlmm'
auto customMlmmArgs(
    const torch::Tensor& atomic_numbers,
    const torch::Tensor& ml_coords,
    const torch::Tensor& mm_coords,
    const torch::Tensor& mm_charges,
    const std::optional<torch::Tensor>& mm_dipoles,
    const std::optional<torch::Tensor>& mm_quads
) -> std::vector<torch::jit::IValue> {
    std::vector<torch::jit::IValue> args = {
        std::tuple{atomic_numbers, ml_coords},
        mm_coords,
        mm_charges,
    };
    if (mm_dipoles.has_value()) {
        args.push_back(*mm_dipoles);
    }
    if (mm_quads.has_value()) {
        args.push_back(*mm_quads);
    }
    return args;
}

// Tuple outputs are (energies, ...) or (species, energies, ...)
auto checkedTupleElements(const torch::jit::IValue& output)
    -> const c10::ivalue::TupleElements& {
    const c10::ivalue::TupleElements& elements = output.toTuple()->elements();
    bool energies_are_first =
        !elements.empty() && elements[0].toTensor().is_floating_point();
    std::size_t min_len = energies_are_first ? 1 : 2;
    if (elements.size() < min_len) {
        throw std::runtime_error(
            "Expected the model to return a tuple (energies, ...) or"
            " (species, energies, ...), but it returned a tuple of length "
            + std::to_string(elements.size())
        );
    }
    return elements;
}

torch::Tensor fetchEnergiesFromOutput(const torch::jit::IValue& output) {
    // Signature of the ML/MM methods that return only the energies
    if (output.isTensor()) {
        return output.toTensor();
    }
    // New signature for compute_from_neighbors
    if (!output.isTuple()) {
        return output.toGenericDict().at("energies").toTensor();
    }

    const c10::ivalue::TupleElements& elements = checkedTupleElements(output);
    torch::Tensor first_tensor = elements[0].toTensor();
    if (first_tensor.is_floating_point()) {
        return first_tensor;
    }

    // Standard signature
    return elements[1].toTensor();
}

// Outputs of the model other than the species and the energies, e.g. "atomic_charges"
auto fetchExtraOutputs(const torch::jit::IValue& output) -> mlip::MlipOutput {
    mlip::MlipOutput extra_outputs;
    if (output.isTensor()) {
        return extra_outputs;
    }
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
    const c10::ivalue::TupleElements& elements = checkedTupleElements(output);
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
    if (output.isGenericDict() or output.isTensor()) {
        return;
    }
    try {
        output.toTuple();
    } catch (const c10::Error& e) {
        std::cerr << "ERROR (mlip_interface)\n"
                  << "Expected model to return a tuple of tensors" << std::endl;
#ifdef DEBUG
        throw;
#endif
        std::exit(2);
    }
    if (output.toTuple()->elements().size() < expect_len) {
        std::cerr << "ERROR (mlip_interface)\n";
        std::cerr << "Expected model to return ";
        if (expect_len == 2) {
            if (external) {
                std::cerr << "a tuple (energies, atomic_scalars)";
            } else {
                std::cerr << "a tuple (species, energies, ...)";
            }
        } else if (expect_len == 3) {
            std::cerr
                << "a tuple (species, energies, atomic_charges, ...) or (species, energies, scalars)";
        } else {
            std::cerr << "a tuple of len at least " << expect_len;
        }
        std::cerr << std::endl;
        std::exit(2);
    }
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
    : mlip::MlipModel{mlip::ModelFormat::TORCHSCRIPT},
      m_module{torch::jit::load(path, device)},
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

auto TorchScriptMlipModel::setExtraScalarAtomicFeatures(
    torch::Tensor extra_scalar_atomic_features
) -> void {
    if (not m_module.find_method("set_extra_scalar_atomic_features").has_value()) {
        throw std::runtime_error(
            "Model doesn't export a method 'set_extra_scalar_atomic_features(list[float]) -> None'"
        );
    }
    extra_scalar_atomic_features =
        extra_scalar_atomic_features.to(torch::kCPU, torch::kDouble).contiguous();
    auto extra_scalar_atomic_features_a =
        extra_scalar_atomic_features.accessor<double, 1>();
    torch::List<double> features;
    for (std::int64_t i = 0; i != extra_scalar_atomic_features.size(0); ++i) {
        features.push_back(extra_scalar_atomic_features_a[i]);
    }
    m_module.get_method("set_extra_scalar_atomic_features")({features});
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
    } catch (const std::exception& e) {
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
        throw std::runtime_error(
            "computeSinglePoint doesn't support ensemble_values=true"
        );
    }
    torch::jit::Kwargs kwargs = commonKwargs(
        coords,
        net_charge,
        m_legacy_charge_type,
        /* ensemble_values= */ calc_qbc_and_qbc_grad,
        molecule_idxs
    );
    kwargs["cell"] = cell;
    kwargs["pbc"] = pbcFromCell(cell);
    torch::jit::Method method = m_module.get_method("forward");
    insertSpinMultKwarg(method, kwargs, spin_mult, coords);
    torch::jit::IValue raw_output =
        method({std::tuple{atomic_numbers, coords}}, kwargs);
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
        throw std::runtime_error(
            "computeFromExternalNeighbors doesn't support atomic=true"
        );
    }
    if (ensemble_values) {
        throw std::runtime_error(
            "computeFromExternalNeighbors doesn't support ensemble_values=true"
        );
    }
    torch::jit::Kwargs kwargs = commonKwargs(
        coords,
        net_charge,
        m_legacy_charge_type,
        /* ensemble_values= */ calc_qbc_and_qbc_grad,
        molecule_idxs
    );
    torch::jit::Method method = m_module.get_method("compute_from_external_neighbors");
    insertSpinMultKwarg(method, kwargs, spin_mult, coords);
    torch::jit::IValue raw_output =
        method({atomic_numbers, coords, neighborlist, shifts}, kwargs);
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
    torch::jit::Kwargs kwargs = commonKwargs(
        ml_coords,
        net_charge,
        m_legacy_charge_type,
        /* ensemble_values= */ calc_qbc_and_qbc_grad,
        molecule_idxs
    );
    kwargs["cell"] = cell;
    kwargs["pbc"] = pbcFromCell(cell);
    return computeMlmmCoreImpl(
        "compute_mlmm",
        customMlmmArgs(
            atomic_numbers,
            ml_coords,
            mm_coords,
            mm_charges,
            /* mm_dipoles= */ std::nullopt,
            /* mm_quads= */ std::nullopt
        ),
        kwargs,
        ml_coords,
        mm_coords,
        spin_mult,
        mm_cutoff,
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
    torch::jit::Kwargs kwargs = commonKwargs(
        ml_coords,
        net_charge,
        m_legacy_charge_type,
        /* ensemble_values= */ calc_qbc_and_qbc_grad,
        molecule_idxs
    );
    kwargs["cell"] = cell;
    kwargs["pbc"] = pbcFromCell(cell);
    return computeMlmmCoreImpl(
        "compute_multipole_mlmm",
        customMlmmArgs(
            atomic_numbers, ml_coords, mm_coords, mm_charges, mm_dipoles, mm_quads
        ),
        kwargs,
        ml_coords,
        mm_coords,
        spin_mult,
        mm_cutoff,
        atomic,
        ensemble_values,
        calc_forces,
        calc_qbc_and_qbc_grad
    );
}

auto TorchScriptMlipModel::computeMlmmCoreImpl(
    const std::string& method_name,
    std::vector<torch::jit::IValue> args,
    torch::jit::Kwargs kwargs,
    const torch::Tensor& ml_coords,
    const torch::Tensor& mm_coords,
    int spin_mult,
    double mm_cutoff,
    bool atomic,
    bool ensemble_values,
    bool calc_forces,
    bool calc_qbc_and_qbc_grad
) -> mlip::MlipOutput {
    if (not m_module.find_method(method_name).has_value()) {
        throw std::runtime_error(
            "Model doesn't export a method '" + method_name + "(...)'"
        );
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
    };
    const c10::FunctionSchema& schema = method.function().getSchema();
    for (const auto& [name, value, is_default] : extra_inputs) {
        if (schema.argumentIndexWithName(name).has_value()) {
            kwargs[name] = value;
        } else if (not is_default) {
            throw std::runtime_error(
                "The model's '" + method_name + "' has no '" + name
                + "' argument, only its default value is supported"
            );
        }
    }
    torch::jit::IValue raw_output = method(std::move(args), kwargs);
    return computeOutput(
        fetchEnergiesFromOutput(raw_output),
        fetchExtraOutputs(raw_output),
        ml_coords,
        mm_coords,
        calc_forces,
        calc_qbc_and_qbc_grad
    );
}

}  // namespace torchscript
