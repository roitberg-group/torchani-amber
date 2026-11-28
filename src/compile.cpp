#include "compile.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <caffe2/serialize/inline_container.h>
#include <torch/all.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include "mlip.h"

namespace {
auto deviceAndDtypeName(torch::Device device, torch::ScalarType dtype) -> std::string {
    std::string device_name;
    if (device.is_cuda()) {
        device_name = "cuda";
    } else if (device.is_cpu()) {
        device_name = "cpu";
    } else {
        throw std::runtime_error("Compiled models only support CPU and CUDA devices");
    }
    std::string dtype_name;
    if (dtype == torch::kFloat64) {
        dtype_name = "f64";
    } else if (dtype == torch::kFloat32) {
        dtype_name = "f32";
    } else {
        throw std::runtime_error("Compiled models only support float32 and float64");
    }
    return device_name + "/" + dtype_name;
}

auto packageHasModel(const std::string& path, const std::string& model_name) -> bool {
    caffe2::serialize::PyTorchStreamReader reader{path};
    std::string model_dir = "data/aotinductor/" + model_name + "/";
    for (const std::string& record : reader.getAllRecords()) {
        if (record.rfind(model_dir, 0) == 0) {
            return true;
        }
    }
    return false;
}
}  // namespace

namespace compile {

CompiledMlipModel::CompiledMlipModel(
    const std::string& path, torch::Device device, torch::ScalarType dtype
)
    : mlip::MlipModel{mlip::ModelFormat::PT2_AOTI},
      m_path{path},
      m_device{device},
      m_dtype{dtype},
      m_supports_external_neighbors{
          packageHasModel(path, modelName("external_neighbors/"))
      },
      m_supports_custom_mlmm{packageHasModel(path, modelName("custom_mlmm/"))},
      m_supports_multipole_mlmm{packageHasModel(path, modelName("multipole_mlmm/"))} {}

auto CompiledMlipModel::modelName(const std::string& prefix) const -> std::string {
    return prefix + deviceAndDtypeName(m_device, m_dtype);
}

auto CompiledMlipModel::makeLoader(const std::string& prefix) const
    -> std::unique_ptr<Loader> {
    return std::make_unique<Loader>(
        m_path,
        modelName(prefix),
        /* run_single_threaded= */ false,
        /* num_runners= */ 1,
        // The CPU runner only accepts index -1
        m_device.is_cpu() ? -1 : m_device.index()
    );
}

auto CompiledMlipModel::perMoleculeTensor(int value, std::int64_t num_molecules) const
    -> torch::Tensor {
    return torch::full(
        {num_molecules},
        value,
        torch::TensorOptions().dtype(torch::kLong).device(m_device)
    );
}

auto CompiledMlipModel::useOnlyEnsembleMember(std::int64_t /* member_idx */) -> void {
    throw std::runtime_error(
        "Compiled models don't support selecting ensemble members"
    );
}

auto CompiledMlipModel::setExtraFeatures(torch::Tensor /* extra_features */) -> void {
    throw std::runtime_error("Compiled models don't support extra features");
}

// Same as TorchScript models that don't export 'set_strategy', this is a no-op
auto CompiledMlipModel::setStrategy(const std::string& /* strat */) -> void {}

auto CompiledMlipModel::supportsCustomMlmm() const -> bool {
    return m_supports_custom_mlmm;
}

auto CompiledMlipModel::supportsMultipoleMlmm() const -> bool {
    return m_supports_multipole_mlmm;
}

auto CompiledMlipModel::supportsComputeFromExternalNeighbors() const -> bool {
    return m_supports_external_neighbors;
}

auto CompiledMlipModel::computeSinglePoint(
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
        throw std::runtime_error("Compiled models don't support atomic=true");
    }
    if (ensemble_values) {
        throw std::runtime_error("Compiled models don't support ensemble_values=true");
    }
    if (molecule_idxs.has_value()) {
        throw std::runtime_error("Compiled models don't support molecule_idxs");
    }
    if (calc_qbc_and_qbc_grad) {
        throw std::runtime_error(
            "Compiled models don't support calc_qbc_and_qbc_grad=true"
        );
    }
    std::vector<torch::Tensor> raw_output;
    if (cell.has_value()) {
        if (not m_pbc_single_point_loader) {
            m_pbc_single_point_loader = makeLoader("single_point_pbc/");
        }
        torch::Tensor pbc = torch::tensor(
            {true, true, true},
            torch::TensorOptions().dtype(torch::kBool).device(cell->device())
        );
        raw_output = m_pbc_single_point_loader->run(
            {atomic_numbers,
                coords,
                *cell,
                pbc,
                perMoleculeTensor(net_charge, coords.size(0)),
                perMoleculeTensor(spin_mult, coords.size(0))}
        );
    } else {
        if (not m_single_point_loader) {
            m_single_point_loader = makeLoader("single_point/");
        }
        raw_output = m_single_point_loader->run(
            {atomic_numbers,
                coords,
                perMoleculeTensor(net_charge, coords.size(0)),
                perMoleculeTensor(spin_mult, coords.size(0))}
        );
    }
    mlip::MlipOutput output;
    output["energy"] = raw_output.at(0);
    if (calc_forces) {
        output["forces"] = raw_output.at(1);
    }
    // Models that don't predict atomic charges return an empty tensor
    if (raw_output.size() >= 3 and raw_output[2].numel() > 0) {
        output["atomic_charges"] = raw_output[2];
    }
    return output;
}

auto CompiledMlipModel::computeFromExternalNeighbors(
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
    if (not m_supports_external_neighbors) {
        throw std::runtime_error(
            "Package doesn't have a model '" + modelName("external_neighbors/") + "'"
        );
    }
    if (atomic) {
        throw std::runtime_error("Compiled models don't support atomic=true");
    }
    if (ensemble_values) {
        throw std::runtime_error("Compiled models don't support ensemble_values=true");
    }
    if (molecule_idxs.has_value()) {
        throw std::runtime_error("Compiled models don't support molecule_idxs");
    }
    if (calc_qbc_and_qbc_grad) {
        throw std::runtime_error(
            "Compiled models don't support calc_qbc_and_qbc_grad=true"
        );
    }
    if (not m_external_neighbors_loader) {
        m_external_neighbors_loader = makeLoader("external_neighbors/");
    }
    std::vector<torch::Tensor> raw_output = m_external_neighbors_loader->run(
        {atomic_numbers,
            coords,
            neighborlist,
            shifts,
            perMoleculeTensor(net_charge, coords.size(0)),
            perMoleculeTensor(spin_mult, coords.size(0))}
    );
    mlip::MlipOutput output;
    output["energy"] = raw_output.at(0);
    if (calc_forces) {
        output["forces"] = raw_output.at(1);
    }
    // Models that don't predict atomic charges return an empty tensor
    if (raw_output.size() >= 3 and raw_output[2].numel() > 0) {
        output["atomic_charges"] = raw_output[2];
    }
    return output;
}

auto CompiledMlipModel::computeMlmm(
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
    std::vector<torch::Tensor> inputs = {
        atomic_numbers,
        ml_coords,
        mm_coords,
        mm_charges,
        perMoleculeTensor(net_charge, ml_coords.size(0)),
        perMoleculeTensor(spin_mult, ml_coords.size(0)),
        torch::full({ml_coords.size(0)}, mm_cutoff, ml_coords.options()),
    };
    return runMlmmLoader(
        m_custom_mlmm_loader,
        "custom_mlmm/",
        m_supports_custom_mlmm,
        inputs,
        cell,
        atomic,
        ensemble_values,
        molecule_idxs,
        calc_forces,
        calc_qbc_and_qbc_grad
    );
}

auto CompiledMlipModel::computeMultipoleMlmm(
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
    std::vector<torch::Tensor> inputs = {
        atomic_numbers,
        ml_coords,
        mm_coords,
        mm_charges,
        mm_dipoles,
        mm_quads,
        perMoleculeTensor(net_charge, ml_coords.size(0)),
        perMoleculeTensor(spin_mult, ml_coords.size(0)),
        torch::full({ml_coords.size(0)}, mm_cutoff, ml_coords.options()),
    };
    return runMlmmLoader(
        m_multipole_mlmm_loader,
        "multipole_mlmm/",
        m_supports_multipole_mlmm,
        inputs,
        cell,
        atomic,
        ensemble_values,
        molecule_idxs,
        calc_forces,
        calc_qbc_and_qbc_grad
    );
}

auto CompiledMlipModel::runMlmmLoader(
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
) -> mlip::MlipOutput {
    if (not package_has_model) {
        throw std::runtime_error(
            "Package doesn't have a model '" + modelName(prefix) + "'"
        );
    }
    if (atomic) {
        throw std::runtime_error("Compiled models don't support atomic=true");
    }
    if (ensemble_values) {
        throw std::runtime_error("Compiled models don't support ensemble_values=true");
    }
    if (cell.has_value()) {
        throw std::runtime_error(
            "Compiled models don't support PBC in ML/MM computations"
        );
    }
    if (molecule_idxs.has_value()) {
        throw std::runtime_error("Compiled models don't support molecule_idxs");
    }
    if (calc_qbc_and_qbc_grad) {
        throw std::runtime_error(
            "Compiled models don't support calc_qbc_and_qbc_grad=true"
        );
    }
    if (not loader) {
        loader = makeLoader(prefix);
    }
    std::vector<torch::Tensor> raw_output = loader->run(inputs);
    mlip::MlipOutput output;
    output["energy"] = raw_output.at(0);
    if (calc_forces) {
        output["forces"] = raw_output.at(1);
        output["mm_forces"] = raw_output.at(2);
    }
    // Models that don't predict atomic charges return an empty tensor
    if (raw_output.at(3).numel() > 0) {
        output["atomic_charges"] = raw_output.at(3);
    }
    return output;
}

}  // namespace compile
