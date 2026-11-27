#include "python_model.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/embed.h>
#include <torch/all.h>
#include <torch/csrc/utils/pybind.h>

#include "mlip.h"
#include "model_output.h"

namespace py = pybind11;
namespace fs = std::filesystem;

namespace {

// After this, no thread holds the GIL, every call into Python acquires it. The
// interpreter is never finalized, since torch can't be re-imported after that
auto ensurePythonInterpreter() -> void {
    if (Py_IsInitialized()) {
        return;
    }
    // Without the program dir, pybind11 would add the cwd (the md run dir) to sys.path
    py::initialize_interpreter(
        /* init_signal_handlers= */ false,
        /* argc= */ 0,
        /* argv= */ nullptr,
        /* add_program_dir_to_path= */ false
    );
    PyEval_SaveThread();
}

// Name under which a sibling file or dir of model.py would be imported, or "" if none
auto importableName(const fs::directory_entry& entry) -> std::string {
    std::string filename = entry.path().filename().string();
    if (entry.is_directory() and fs::is_regular_file(entry.path() / "__init__.py")) {
        return filename;
    }
    bool is_module_file =
        entry.path().extension() == ".py" or entry.path().extension() == ".so";
    if (entry.is_regular_file() and is_module_file) {
        return filename.substr(0, filename.find('.'));
    }
    return "";
}

// Throws if a module importable from model_dir is already importable from elsewhere,
// since it would either shadow, or be shadowed by, the other module
auto checkSiblingNamesAreUnique(const fs::path& model_dir) -> void {
    py::object find_spec = py::module_::import("importlib.util").attr("find_spec");
    std::string dir_prefix = model_dir.string() + "/";
    for (const fs::directory_entry& entry : fs::directory_iterator(model_dir)) {
        std::string name = importableName(entry);
        if (name.empty() or name == "model"
            or not py::str(name).attr("isidentifier")().cast<bool>()) {
            continue;
        }
        py::object spec = find_spec(name);
        if (spec.is_none()) {
            continue;
        }
        py::object origin = spec.attr("origin");
        if (not origin.is_none()
            and origin.cast<std::string>().rfind(dir_prefix, 0) == 0) {
            continue;
        }
        throw std::runtime_error(
            "Module '" + name + "' in " + model_dir.string()
            + " clashes with an already importable module of the same name."
            " Python model dirs must have files with names unique among all loaded"
            " Python models and installed Python packages"
        );
    }
}

auto flushPythonStdout() -> void {
    // The interpreter is never finalized, so buffered output would be lost at exit
    py::module_::import("sys").attr("stdout").attr("flush")();
}

auto methodAcceptsArg(const py::object& method, const char* name) -> bool {
    py::object signature = py::module_::import("inspect").attr("signature")(method);
    py::object parameters = signature.attr("parameters");
    if (parameters.contains(name)) {
        return true;
    }
    py::object var_keyword =
        py::module_::import("inspect").attr("Parameter").attr("VAR_KEYWORD");
    for (py::handle parameter : parameters.attr("values")()) {
        py::object kind = parameter.attr("kind");
        if (kind.equal(var_keyword)) {
            return true;
        }
    }
    return false;
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

auto perMoleculeLongTensor(int value, const torch::Tensor& coords) -> torch::Tensor {
    return torch::full({coords.size(0)}, value, coords.options().dtype(torch::kLong));
}

// Old models take 'charge: int', new ones 'charge: Tensor | None' of shape (molecules,)
auto chargeArgIsInt(const py::object& model) -> bool {
    py::object forward = model.attr("forward");
    if (not methodAcceptsArg(forward, "charge")) {
        throw std::runtime_error("Model's 'forward' has no 'charge' argument");
    }
    py::object signature = py::module_::import("inspect").attr("signature")(forward);
    py::object parameters = signature.attr("parameters");
    if (not parameters.contains("charge")) {
        return false;
    }
    py::object annotation = parameters["charge"].attr("annotation");
    // Annotations are strings in modules with 'from __future__ import annotations'
    if (py::isinstance<py::str>(annotation)) {
        return annotation.cast<std::string>() == "int";
    }
    return annotation.is(py::module_::import("builtins").attr("int"));
}

auto chargeInput(int net_charge, const torch::Tensor& coords, bool legacy_charge_type)
    -> py::object {
    if (legacy_charge_type) {
        return py::int_(net_charge);
    }
    return py::cast(perMoleculeLongTensor(net_charge, coords));
}

// Models whose method has no 'spin_mult' argument only support spin_mult = 1
auto insertSpinMultKwarg(
    const py::object& method,
    const char* method_name,
    py::dict& kwargs,
    int spin_mult,
    const torch::Tensor& coords
) -> void {
    if (methodAcceptsArg(method, "spin_mult")) {
        kwargs["spin_mult"] = perMoleculeLongTensor(spin_mult, coords);
    } else if (spin_mult != 1) {
        throw std::runtime_error(
            "The model's '" + std::string{method_name}
            + "' has no 'spin_mult' argument, only spin_mult = 1 is supported"
        );
    }
}

// The energies and the extra outputs (e.g. "atomic_charges") returned by a model method
struct ModelOutput {
    torch::Tensor energies;
    mlip::MlipOutput extra_outputs;
};

auto isTensor(const py::handle& obj) -> bool {
    return py::isinstance(obj, py::module_::import("torch").attr("Tensor"));
}

// Extra outputs are None, a tensor of atomic charges, or a dict of tensors
auto addExtraOutputs(const py::handle& extra, mlip::MlipOutput& extra_outputs) -> void {
    if (isTensor(extra)) {
        extra_outputs["atomic_charges"] = extra.cast<torch::Tensor>();
        return;
    }
    if (not py::isinstance<py::dict>(extra)) {
        return;
    }
    for (auto [key, value] : extra.cast<py::dict>()) {
        std::string name = key.cast<std::string>();
        if (name != "energies" and isTensor(value)) {
            extra_outputs[name] = value.cast<torch::Tensor>();
        }
    }
}

// The output is the energies, a dict with "energies" and extra outputs, or a tuple
// (energies, extra) or (species, energies, extra), where extra is optional
auto parseModelOutput(const py::object& output, const char* method_name)
    -> ModelOutput {
    ModelOutput parsed;
    if (isTensor(output)) {
        parsed.energies = output.cast<torch::Tensor>();
        return parsed;
    }
    if (py::isinstance<py::dict>(output) and output.contains("energies")) {
        parsed.energies = output["energies"].cast<torch::Tensor>();
        addExtraOutputs(output, parsed.extra_outputs);
        return parsed;
    }
    if (py::isinstance<py::tuple>(output) and py::len(output) > 0
        and isTensor(output[py::int_(0)])) {
        py::tuple elements = output.cast<py::tuple>();
        // The species are integers, the energies are floating point
        std::size_t energies_idx =
            elements[0].cast<torch::Tensor>().is_floating_point() ? 0 : 1;
        if (py::len(elements) > energies_idx) {
            parsed.energies = elements[energies_idx].cast<torch::Tensor>();
            if (py::len(elements) > energies_idx + 1) {
                addExtraOutputs(elements[energies_idx + 1], parsed.extra_outputs);
            }
            return parsed;
        }
    }
    throw std::runtime_error(
        "Unexpected output of the model's '" + std::string{method_name}
        + "', expected the energies, a dict with an 'energies' key, a tuple"
          " (energies, ...) or a tuple (species, energies, ...)"
    );
}

// Requires the GIL
auto callMethod(
    const py::object& method,
    const char* method_name,
    const py::tuple& args,
    const py::dict& kwargs
) -> ModelOutput {
    py::object output = method(*args, **kwargs);
    flushPythonStdout();
    return parseModelOutput(output, method_name);
}

auto checkNotAtomicOrEnsemble(
    const std::string& name, bool atomic, bool ensemble_values
) -> void {
    if (atomic) {
        throw std::runtime_error(name + " doesn't support atomic=true");
    }
    if (ensemble_values) {
        throw std::runtime_error(name + " doesn't support ensemble_values=true");
    }
}

}  // namespace

namespace python_model {

PythonMlipModel::PythonMlipModel(
    const std::string& model_dir,
    torch::Device device,
    torch::ScalarType dtype,
    const mlip::ModelOptions& options
)
    : mlip::MlipModel{mlip::ModelFormat::PYTHON} {
    fs::path model_file = fs::path(model_dir) / "model.py";
    if (not fs::is_regular_file(model_file)) {
        throw std::runtime_error("Python model file not found: " + model_file.string());
    }
    fs::path model_dir_path = fs::canonical(model_dir);
    model_file = model_dir_path / "model.py";
    ensurePythonInterpreter();
    py::gil_scoped_acquire gil;
    checkSiblingNamesAreUnique(model_dir_path);
    py::module_ sys = py::module_::import("sys");
    // Lets model.py import sibling modules
    sys.attr("path").attr("insert")(0, model_dir_path.string());
    // Unique per dir, and registered, since e.g. dataclasses and pickle look it up
    std::string module_name = "torchani_user_model_"
        + std::to_string(std::hash<std::string>{}(model_file.string()));
    py::module_ importlib_util = py::module_::import("importlib.util");
    py::object spec = importlib_util.attr("spec_from_file_location")(
        module_name, model_file.string()
    );
    py::object model_module = importlib_util.attr("module_from_spec")(spec);
    sys.attr("modules")[py::str(module_name)] = model_module;
    spec.attr("loader").attr("exec_module")(model_module);

    py::module_ torch_py = py::module_::import("torch");
    py::object dtype_py =
        torch_py.attr(dtype == torch::kDouble ? "float64" : "float32");
    // Declared after the GIL guard, so that it is released while the GIL is held if
    // something below throws
    py::object model = model_module.attr("make_model")(device.str(), dtype_py);
    flushPythonStdout();
    if (not options.skip_py_preamble) {
        model.attr("to")(py::arg("device") = device.str(), py::arg("dtype") = dtype_py);
        model.attr("eval")();
    }
    m_legacy_charge_type = chargeArgIsInt(model);
    if (options.do_torch_compile) {
        const std::array<const char*, 4> compute_methods = {
            "forward",
            "compute_from_external_neighbors",
            "compute_mlmm",
            "compute_multipole_mlmm",
        };
        for (const char* method : compute_methods) {
            if (py::hasattr(model, method)) {
                model.attr(method) = torch_py.attr("compile")(
                    model.attr(method), py::arg("mode") = options.torch_compile_mode
                );
            }
        }
    }
    m_model = std::move(model);
}

PythonMlipModel::~PythonMlipModel() {
    // If the md engine is itself a Python program, it may have already finalized
    if (not Py_IsInitialized()) {
        m_model.release();
        return;
    }
    py::gil_scoped_acquire gil;
    m_model = py::object();
}

auto PythonMlipModel::hasMethod(const char* name) const -> bool {
    py::gil_scoped_acquire gil;
    return py::hasattr(m_model, name);
}

auto PythonMlipModel::useOnlyEnsembleMember(std::int64_t member_idx) -> void {
    if (not hasMethod("set_active_members")) {
        throw std::runtime_error(
            "Model doesn't export a method 'set_active_members(list[int]) -> None'"
        );
    }
    py::gil_scoped_acquire gil;
    py::list idxs;
    idxs.append(member_idx);
    m_model.attr("set_active_members")(idxs);
}

auto PythonMlipModel::setExtraFeatures(torch::Tensor extra_features) -> void {
    if (not hasMethod("set_extra_features")) {
        throw std::runtime_error(
            "Model doesn't export a method 'set_extra_features(list[float]) -> None'"
        );
    }
    extra_features = extra_features.to(torch::kCPU, torch::kDouble).contiguous();
    auto extra_features_a = extra_features.accessor<double, 1>();
    py::gil_scoped_acquire gil;
    py::list features;
    for (std::int64_t i = 0; i != extra_features.size(0); ++i) {
        features.append(extra_features_a[i]);
    }
    m_model.attr("set_extra_features")(features);
}

auto PythonMlipModel::supportsCustomMlmm() const -> bool {
    return hasMethod("compute_mlmm");
}

auto PythonMlipModel::supportsMultipoleMlmm() const -> bool {
    return hasMethod("compute_multipole_mlmm");
}

auto PythonMlipModel::supportsComputeFromExternalNeighbors() const -> bool {
    return hasMethod("compute_from_external_neighbors");
}

auto PythonMlipModel::setStrategy(const std::string& strat) -> void {
    // Models are not required to export set_strategy
    if (not hasMethod("set_strategy")) {
        return;
    }
    py::gil_scoped_acquire gil;
    try {
        m_model.attr("set_strategy")(strat);
    } catch (const py::error_already_set& e) {
        throw std::runtime_error(
            "The selected model exports 'set_strategy(str) -> None'"
            " but it doesn't accept the specified strategy"
        );
    }
}

auto PythonMlipModel::computeSinglePoint(
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
    checkNotAtomicOrEnsemble("computeSinglePoint", atomic, ensemble_values);
    ModelOutput output;
    {
        py::gil_scoped_acquire gil;
        py::object method = m_model.attr("forward");
        py::dict kwargs;
        kwargs["cell"] = cell;
        kwargs["pbc"] = pbcFromCell(cell);
        kwargs["charge"] = chargeInput(net_charge, coords, m_legacy_charge_type);
        kwargs["atomic"] = false;
        kwargs["ensemble_values"] = calc_qbc_and_qbc_grad;
        kwargs["molecule_idxs"] = molecule_idxs;
        insertSpinMultKwarg(method, "forward", kwargs, spin_mult, coords);
        output = callMethod(
            method,
            "forward",
            py::make_tuple(py::make_tuple(atomic_numbers, coords)),
            kwargs
        );
    }
    return model_output::computeOutput(
        output.energies,
        std::move(output.extra_outputs),
        coords,
        /* mm_coords= */ std::nullopt,
        calc_forces,
        calc_qbc_and_qbc_grad
    );
}

auto PythonMlipModel::computeFromExternalNeighbors(
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
    checkNotAtomicOrEnsemble("computeFromExternalNeighbors", atomic, ensemble_values);
    ModelOutput output;
    {
        py::gil_scoped_acquire gil;
        py::object method = m_model.attr("compute_from_external_neighbors");
        py::dict kwargs;
        kwargs["charge"] = chargeInput(net_charge, coords, m_legacy_charge_type);
        kwargs["atomic"] = false;
        kwargs["ensemble_values"] = calc_qbc_and_qbc_grad;
        kwargs["molecule_idxs"] = molecule_idxs;
        insertSpinMultKwarg(
            method, "compute_from_external_neighbors", kwargs, spin_mult, coords
        );
        output = callMethod(
            method,
            "compute_from_external_neighbors",
            py::make_tuple(atomic_numbers, coords, neighborlist, shifts),
            kwargs
        );
    }
    return model_output::computeOutput(
        output.energies,
        std::move(output.extra_outputs),
        coords,
        /* mm_coords= */ std::nullopt,
        calc_forces,
        calc_qbc_and_qbc_grad
    );
}

auto PythonMlipModel::computeMlmm(
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
    return computeMlmmCoreImpl(
        "compute_mlmm",
        atomic_numbers,
        ml_coords,
        mm_coords,
        mm_charges,
        /* mm_dipoles= */ std::nullopt,
        /* mm_quads= */ std::nullopt,
        cell,
        net_charge,
        spin_mult,
        mm_cutoff,
        atomic,
        ensemble_values,
        molecule_idxs,
        calc_forces,
        calc_qbc_and_qbc_grad
    );
}

auto PythonMlipModel::computeMultipoleMlmm(
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
    return computeMlmmCoreImpl(
        "compute_multipole_mlmm",
        atomic_numbers,
        ml_coords,
        mm_coords,
        mm_charges,
        mm_dipoles,
        mm_quads,
        cell,
        net_charge,
        spin_mult,
        mm_cutoff,
        atomic,
        ensemble_values,
        molecule_idxs,
        calc_forces,
        calc_qbc_and_qbc_grad
    );
}

auto PythonMlipModel::computeMlmmCoreImpl(
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
) -> mlip::MlipOutput {
    if (not hasMethod(method_name)) {
        throw std::runtime_error(
            "Model doesn't export a method '" + std::string{method_name} + "(...)'"
        );
    }
    checkNotAtomicOrEnsemble(method_name, atomic, ensemble_values);
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
    ModelOutput output;
    {
        py::gil_scoped_acquire gil;
        py::object method = m_model.attr(method_name);
        // mm_dipoles and mm_quads are only passed to 'compute_multipole_mlmm'
        py::list args;
        args.append(py::make_tuple(atomic_numbers, ml_coords));
        args.append(mm_coords);
        args.append(mm_charges);
        if (mm_dipoles.has_value()) {
            args.append(*mm_dipoles);
        }
        if (mm_quads.has_value()) {
            args.append(*mm_quads);
        }
        py::dict kwargs;
        kwargs["cell"] = cell;
        kwargs["pbc"] = pbcFromCell(cell);
        kwargs["charge"] = chargeInput(net_charge, ml_coords, m_legacy_charge_type);
        kwargs["atomic"] = false;
        kwargs["ensemble_values"] = calc_qbc_and_qbc_grad;
        kwargs["molecule_idxs"] = molecule_idxs;
        for (const auto& [name, value, is_default] : extra_inputs) {
            if (methodAcceptsArg(method, name.c_str())) {
                kwargs[py::str(name)] = value;
            } else if (not is_default) {
                throw std::runtime_error(
                    "The model's '" + std::string{method_name} + "' has no '" + name
                    + "' argument, only its default value is supported"
                );
            }
        }
        output = callMethod(method, method_name, py::tuple(args), kwargs);
    }
    return model_output::computeOutput(
        output.energies,
        std::move(output.extra_outputs),
        ml_coords,
        mm_coords,
        calc_forces,
        calc_qbc_and_qbc_grad
    );
}

}  // namespace python_model
