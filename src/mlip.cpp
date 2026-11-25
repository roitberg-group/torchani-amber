#include "mlip.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <caffe2/serialize/inline_container.h>
#include <torch/all.h>

#include "compile.h"
#include "torchscript.h"

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
        return mlip::ModelFormat::PYTHON;
    }
    return mlip::ModelFormat::TORCHSCRIPT;
}
}  // namespace

namespace mlip {

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

auto MlipModel::from_python_model_path(
    const std::string& /* path */,
    torch::Device /* device */,
    torch::ScalarType /* dtype */
) -> std::unique_ptr<MlipModel> {
    throw std::runtime_error(
        "Loading Python models saved with torch.save(model) is not supported"
    );
}

auto MlipModel::from_model_path(
    const std::string& path,
    torch::Device device,
    torch::ScalarType dtype,
    ModelFormat format
) -> std::unique_ptr<MlipModel> {
    ModelFormat resolved_format = format;
    if (format == ModelFormat::AUTO) {
        resolved_format = detectModelFormat(path);
    }
    if (resolved_format == ModelFormat::PYTHON) {
        return from_python_model_path(path, device, dtype);
    }
    // Only .pt2 packages with AOTInductor-compiled models are supported. A .pt2
    // that holds plain exported programs can't be loaded
    if (resolved_format == ModelFormat::PT2_AOTI) {
        return from_pt2_aoti_model_path(path, device, dtype);
    }
    return from_jit_model_path(path, device, dtype);
}

}  // namespace mlip
