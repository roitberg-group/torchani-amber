#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include <torch/all.h>

namespace mlip {

// Keys: "energy", "forces", "qbc", "qbc_grad", "mm_forces", and the extra outputs of
// the model, if any (e.g. "atomic_charges"). All values are in Hartree and Angstrom.
using MlipOutput = std::unordered_map<std::string, torch::Tensor>;

enum class ModelFormat { TORCHSCRIPT, PT2_AOTI, PYTHON, AUTO };

// Accepts "torchscript", "pt2-aoti", "python" and "auto"
auto modelFormatFromString(const std::string& format) -> ModelFormat;
auto modelFormatToString(ModelFormat format) -> std::string;

// Only used by Python models, other formats ignore them with a warning
struct ModelOptions {
    bool skip_py_preamble = false;
    bool do_torch_compile = false;
    std::string torch_compile_mode = "default";
};

/**
 * Interface for a machine learning interatomic potential
 * */
class MlipModel {
    const ModelFormat m_format;

    protected:
    explicit MlipModel(ModelFormat format);

    public:
    static auto from_jit_model_path(
        const std::string& path, torch::Device device, torch::ScalarType dtype
    ) -> std::unique_ptr<MlipModel>;
    static auto from_pt2_aoti_model_path(
        const std::string& path, torch::Device device, torch::ScalarType dtype
    ) -> std::unique_ptr<MlipModel>;
    static auto from_python_model_path(
        const std::string& path,
        torch::Device device,
        torch::ScalarType dtype,
        const ModelOptions& options = {}
    ) -> std::unique_ptr<MlipModel>;
    static auto from_model_path(
        const std::string& path,
        torch::Device device,
        torch::ScalarType dtype,
        const ModelOptions& options = {},
        ModelFormat format = ModelFormat::AUTO
    ) -> std::unique_ptr<MlipModel>;

    // The format the model was loaded as, never AUTO
    auto format() const -> ModelFormat;

    virtual ~MlipModel() = default;

    virtual auto useOnlyEnsembleMember(std::int64_t member_idx) -> void = 0;
    virtual auto setExtraScalarAtomicFeatures(
        torch::Tensor extra_scalar_atomic_features
    ) -> void = 0;
    virtual auto setStrategy(const std::string& strat) -> void = 0;
    virtual auto supportsCustomMlmm() const -> bool = 0;
    virtual auto supportsMultipoleMlmm() const -> bool = 0;
    virtual auto supportsComputeFromExternalNeighbors() const -> bool = 0;
    virtual auto computeSinglePoint(
        const torch::Tensor& atomic_numbers,
        const torch::Tensor& coords,
        const std::optional<torch::Tensor>& cell = std::nullopt,
        int net_charge = 0,
        int spin_mult = 1,
        bool atomic = false,
        bool ensemble_values = false,
        const std::optional<torch::Tensor>& molecule_idxs = std::nullopt,
        bool calc_forces = false,
        bool calc_qbc_and_qbc_grad = false
    ) -> MlipOutput = 0;
    virtual auto computeFromExternalNeighbors(
        const torch::Tensor& atomic_numbers,
        const torch::Tensor& coords,
        const torch::Tensor& neighborlist,
        const torch::Tensor& shifts,
        int net_charge = 0,
        int spin_mult = 1,
        bool atomic = false,
        bool ensemble_values = false,
        const std::optional<torch::Tensor>& molecule_idxs = std::nullopt,
        bool calc_forces = false,
        bool calc_qbc_and_qbc_grad = false
    ) -> MlipOutput = 0;
    virtual auto computeMlmm(
        const torch::Tensor& atomic_numbers,
        const torch::Tensor& ml_coords,
        const torch::Tensor& mm_coords,
        const torch::Tensor& mm_charges,
        const std::optional<torch::Tensor>& cell = std::nullopt,
        int net_charge = 0,
        int spin_mult = 1,
        double mm_cutoff = 0.0,
        bool atomic = false,
        bool ensemble_values = false,
        const std::optional<torch::Tensor>& molecule_idxs = std::nullopt,
        bool calc_forces = false,
        bool calc_qbc_and_qbc_grad = false
    ) -> MlipOutput = 0;
    virtual auto computeMultipoleMlmm(
        const torch::Tensor& atomic_numbers,
        const torch::Tensor& ml_coords,
        const torch::Tensor& mm_coords,
        const torch::Tensor& mm_charges,
        const torch::Tensor& mm_dipoles,
        const torch::Tensor& mm_quads,
        const std::optional<torch::Tensor>& cell = std::nullopt,
        int net_charge = 0,
        int spin_mult = 1,
        double mm_cutoff = 0.0,
        bool atomic = false,
        bool ensemble_values = false,
        const std::optional<torch::Tensor>& molecule_idxs = std::nullopt,
        bool calc_forces = false,
        bool calc_qbc_and_qbc_grad = false
    ) -> MlipOutput = 0;
};

}  // namespace mlip
