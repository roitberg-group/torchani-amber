#include "python_model.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <functional>
#include <stdexcept>

#include <pybind11/embed.h>
#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;
namespace fs = std::filesystem;

namespace {
// Name under which a sibling file or dir of model.py would be imported, or "" if none
auto importableName(const fs::directory_entry& entry) -> std::string {
    std::string filename = entry.path().filename().string();
    if (entry.is_directory() and fs::is_regular_file(entry.path() / "__init__.py")) {
        return filename;
    }
    if (entry.is_regular_file()
        and (entry.path().extension() == ".py" or entry.path().extension() == ".so")) {
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
}  // namespace

PythonModel::PythonModel(
    const std::string& model_dir,
    torch::Device device,
    torch::ScalarType dtype,
    const ModelOptions& options
) {
    const std::array<std::string, 4> compile_modes = {
        "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
    };
    if (std::find(
            compile_modes.begin(), compile_modes.end(), options.torch_compile_mode
        )
        == compile_modes.end()) {
        throw std::runtime_error(
            "torch_compile_mode must be 'default', 'reduce-overhead', 'max-autotune' or"
            " 'max-autotune-no-cudagraphs', but it is '"
            + options.torch_compile_mode + "'"
        );
    }
    if (not options.do_torch_compile and options.torch_compile_mode != "default") {
        throw std::runtime_error("torch_compile_mode requires do_torch_compile");
    }
    // The interpreter is never finalized, since torch can't be re-imported after that
    if (not Py_IsInitialized()) {
        py::initialize_interpreter(/* init_signal_handlers= */ false);
    }
    fs::path model_dir_path = fs::canonical(model_dir);
    std::string model_file = (model_dir_path / "model.py").string();
    if (not fs::is_regular_file(model_file)) {
        throw std::runtime_error("Python model file not found: " + model_file);
    }
    // If the sibiling names are the same as the ones from other imported model in the
    // same process, python will shadow the imported modules. To avoid this, ensure all
    // sibiling names are unique.
    checkSiblingNamesAreUnique(model_dir_path);
    py::module_ sys = py::module_::import("sys");
    // Lets model.py import sibling modules
    sys.attr("path").attr("insert")(0, model_dir_path.string());
    // Unique per dir, and registered, since e.g. dataclasses and pickle look it up
    std::string module_name =
        "torchani_user_model_" + std::to_string(std::hash<std::string>{}(model_file));
    py::module_ importlib_util = py::module_::import("importlib.util");
    py::object spec =
        importlib_util.attr("spec_from_file_location")(module_name, model_file);
    py::object model_module = importlib_util.attr("module_from_spec")(spec);
    sys.attr("modules")[py::str(module_name)] = model_module;
    spec.attr("loader").attr("exec_module")(model_module);

    py::module_ torch_py = py::module_::import("torch");
    py::object dtype_py =
        torch_py.attr(dtype == torch::kDouble ? "float64" : "float32");
    m_model = model_module.attr("make_model")(device.str(), dtype_py);
    sys.attr("stdout").attr("flush")();
    if (not options.skip_py_preamble) {
        m_model.attr("to")(
            py::arg("device") = device.str(), py::arg("dtype") = dtype_py
        );
        m_model.attr("eval")();
    }
    if (options.do_torch_compile) {
        const std::array<const char*, 4> compute_methods = {
            "forward",
            "compute_from_external_neighbors",
            "compute_mlmm",
            "compute_multipole_mlmm",
        };
        for (const char* method : compute_methods) {
            if (py::hasattr(m_model, method)) {
                m_model.attr(method) = torch_py.attr("compile")(
                    m_model.attr(method), py::arg("mode") = options.torch_compile_mode
                );
            }
        }
    }
}

auto PythonModel::computeEnergyAndForces(
    const torch::Tensor& atomic_numbers, const torch::Tensor& coords
) -> std::pair<torch::Tensor, torch::Tensor> {
    torch::Tensor coords_grad = coords.detach().requires_grad_(true);
    py::tuple species_coords =
        py::make_tuple(atomic_numbers.unsqueeze(0), coords_grad.unsqueeze(0));
    py::tuple output = m_model(species_coords);
    // The interpreter is never finalized, so buffered output would be lost at exit
    py::module_::import("sys").attr("stdout").attr("flush")();
    torch::Tensor energy = output[1].cast<torch::Tensor>();
    torch::Tensor forces;
    {
        // Backward may run Python code (e.g. autograd.Function) in other threads
        py::gil_scoped_release no_gil;
        forces = -torch::autograd::grad({energy.sum()}, {coords_grad})[0];
    }
    return {energy.detach(), forces};
}
