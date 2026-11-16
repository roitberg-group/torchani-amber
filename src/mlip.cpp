#include "mlip.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <c10/util/Logging.h>
#include <torch/all.h>
#include <torch/script.h>

MlipModel::MlipModel(
    const std::string& path, torch::Device device, torch::ScalarType dtype
)
    : m_module{torch::jit::load(path, device)} {
    cast_float_tensors(dtype);
}

// NOTE: Unfortunately, due to a strange LibTorch design choice, Module::to() converts
// *all params and bufs* to the dtype, not only the float tensors like the Python API.
// Here we loop over the params and buffers manually and convert only the floating
// point ones
auto MlipModel::cast_float_tensors(torch::ScalarType dtype) -> void {
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

auto MlipModel::use_only_ensemble_member(std::int64_t member_idx) -> void {
    if (not has_method("set_active_members")) {
        throw std::runtime_error(
            "Model doesn't export a method 'set_active_members(list[int]) -> None'"
        );
    }
    call_method("set_active_members", {torch::List<std::int64_t>{member_idx}});
}

auto MlipModel::read_extra_features_from_file(const std::string& fpath) -> void {
    if (not has_method("set_extra_features")) {
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
    call_method("set_extra_features", {features});
}

auto MlipModel::has_method(const std::string& name) const -> bool {
    return m_module.find_method(name).has_value();
}

auto MlipModel::supports_custom_mlmm() const -> bool { return has_method("compute_mlmm"); }

auto MlipModel::call_method(const std::string& name, std::vector<torch::jit::IValue> inputs) const
    -> torch::jit::IValue {
    return m_module.get_method(name)(std::move(inputs));
}

auto MlipModel::forward(std::vector<torch::jit::IValue> inputs) -> torch::jit::IValue {
    return m_module.forward(std::move(inputs));
}

auto MlipModel::compute_mlmm(std::vector<torch::jit::IValue> inputs) -> torch::jit::IValue {
    if (not supports_custom_mlmm()) {
        throw std::runtime_error("Model doesn't export a method 'compute_mlmm(...)'");
    }
    return call_method("compute_mlmm", std::move(inputs));
}
