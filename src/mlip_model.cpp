#include "mlip_model.h"

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <caffe2/serialize/inline_container.h>
#include <torch/all.h>

#include "compile.h"
#include "torchscript.h"
#include "utils.h"
#ifdef TORCHANI_SUPPORT_PYTHON_MODELS
#include "python_model.h"
#endif

namespace {
auto archiveHasRecordPrefix(
    const std::vector<std::string>& records, const std::string& prefix
) -> bool {
    for (const std::string& record : records) {
        if (record.rfind(prefix, 0) == 0) {
            return true;
        }
    }
    return false;
}

auto detectModelFormat(const std::string& path) -> mlip::ModelFormat {
    if (std::filesystem::is_directory(path)) {
        return mlip::ModelFormat::PYTHON;
    }
    caffe2::serialize::PyTorchStreamReader reader{path};
    std::vector<std::string> records = reader.getAllRecords();
    if (reader.hasRecord("archive_format")) {
        auto [data, size] = reader.getRecord("archive_format");
        std::string archive_format{static_cast<const char*>(data.get()), size};
        if (archive_format == "pt2") {
            if (not archiveHasRecordPrefix(records, "data/aotinductor/")) {
                throw std::runtime_error(
                    "The .pt2 package has no AOTInductor-compiled models."
                    " Plain exported programs can't be loaded"
                );
            }
            return mlip::ModelFormat::PT2_AOTI;
        }
    }
    if (reader.hasRecord("data.pkl") and not reader.hasRecord("constants.pkl")
        and not archiveHasRecordPrefix(records, "code/")) {
        throw std::runtime_error(
            "Models saved with torch.save(model) can't be loaded. Python models must"
            " be a directory with a model.py that defines make_model(device, dtype)"
        );
    }
    return mlip::ModelFormat::TORCHSCRIPT;
}

auto warnIgnoredOptions(const mlip::ModelOptions& options, mlip::ModelFormat format)
    -> void {
    std::string format_name = mlip::modelFormatToString(format);
    if (options.do_torch_compile) {
        utils::printWarning(
            "do_torch_compile and torch_compile_mode only apply to Python models,"
            " they are ignored for the '"
            + format_name + "' model"
        );
    }
    if (options.skip_py_preamble) {
        utils::printWarning(
            "skip_py_preamble only applies to Python models, it is ignored for the '"
            + format_name + "' model"
        );
    }
}
}  // namespace

namespace mlip {

auto modelFormatFromString(const std::string& format) -> ModelFormat {
    if (format == "torchscript") {
        return ModelFormat::TORCHSCRIPT;
    }
    if (format == "pt2-aoti") {
        return ModelFormat::PT2_AOTI;
    }
    if (format == "python") {
        return ModelFormat::PYTHON;
    }
    if (format == "auto") {
        return ModelFormat::AUTO;
    }
    throw std::invalid_argument(
        "Model format must be 'auto', 'torchscript', 'pt2-aoti' or 'python', but it is '"
        + format + "'"
    );
}

auto modelFormatToString(ModelFormat format) -> std::string {
    switch (format) {
        case ModelFormat::TORCHSCRIPT:
            return "torchscript";
        case ModelFormat::PT2_AOTI:
            return "pt2-aoti";
        case ModelFormat::PYTHON:
            return "python";
        case ModelFormat::AUTO:
            return "auto";
    }
    throw std::logic_error("Unknown model format");
}

MlipModel::MlipModel(ModelFormat format) : m_format{format} {}

auto MlipModel::format() const -> ModelFormat {
    return m_format;
}

auto MlipModel::from_jit_model_path(
    const std::string& path, torch::Device device, torch::ScalarType dtype
) -> std::unique_ptr<MlipModel> {
    // Disable broken TorchScript "optimizations"
    torchscript::disableTorchscriptExecutorOptimizations();
    return std::make_unique<torchscript::TorchScriptMlipModel>(path, device, dtype);
}

auto MlipModel::from_pt2_aoti_model_path(
    const std::string& path, torch::Device device, torch::ScalarType dtype
) -> std::unique_ptr<MlipModel> {
    return std::make_unique<compile::CompiledMlipModel>(path, device, dtype);
}

#ifdef TORCHANI_SUPPORT_PYTHON_MODELS
auto MlipModel::from_python_model_path(
    const std::string& path,
    torch::Device device,
    torch::ScalarType dtype,
    const ModelOptions& options
) -> std::unique_ptr<MlipModel> {
    return std::make_unique<python_model::PythonMlipModel>(
        path, device, dtype, options
    );
}
#else
auto MlipModel::from_python_model_path(
    const std::string& /* path */,
    torch::Device /* device */,
    torch::ScalarType /* dtype */,
    const ModelOptions& /* options */
) -> std::unique_ptr<MlipModel> {
    throw std::runtime_error(
        "Python models are not supported, since libtorchani was built with"
        " TORCHANI_SUPPORT_PYTHON_MODELS=OFF"
    );
}
#endif

auto MlipModel::from_model_path(
    const std::string& path,
    torch::Device device,
    torch::ScalarType dtype,
    const ModelOptions& options,
    ModelFormat format
) -> std::unique_ptr<MlipModel> {
    ModelFormat resolved_format = format;
    if (format == ModelFormat::AUTO) {
        resolved_format = detectModelFormat(path);
    }
    if (resolved_format == ModelFormat::PYTHON) {
        return from_python_model_path(path, device, dtype, options);
    }
    warnIgnoredOptions(options, resolved_format);
    // Only .pt2 packages with AOTInductor-compiled models are supported. A .pt2
    // that holds plain exported programs can't be loaded
    if (resolved_format == ModelFormat::PT2_AOTI) {
        return from_pt2_aoti_model_path(path, device, dtype);
    }
    return from_jit_model_path(path, device, dtype);
}

}  // namespace mlip
