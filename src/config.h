#pragma once

#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <c10/core/ScalarType.h>
#include <torch/all.h>
#include <torch/types.h>

#include "fortran_utils.h"

namespace config {

// A namelist option: its value, and the type the namelist parser checks it against
// TODO: The type is redundant, it could be derived from the value with
// std::holds_alternative, and ConfigDict could map keys directly to values
struct ConfigOption {
    // Scalars have exactly one value, VEC_* and VEC3_* any number
    std::vector<fortran_utils::NamelistValue> values;
    fortran_utils::NamelistType type;

    ConfigOption(
        fortran_utils::NamelistValue option_value, fortran_utils::NamelistType option_type
    ):
        values{std::move(option_value)},
        type{option_type}
    {}

    ConfigOption(
        std::vector<fortran_utils::NamelistValue> option_values,
        fortran_utils::NamelistType option_type
    ):
        values{std::move(option_values)},
        type{option_type}
    {}
};

// Thin wrapper over a map of (lowercase) namelist keys to options. The getters throw
// if the key is missing, or if its value has a different type
class ConfigDict {
    std::map<std::string, ConfigOption> m_options;

    public:
    explicit ConfigDict(std::map<std::string, ConfigOption> options):
        m_options{std::move(options)}
    {}

    auto options() const -> const std::map<std::string, ConfigOption>& {
        return m_options;
    }

    auto hasOption(const std::string& key) const -> bool {
        return m_options.count(key) > 0;
    }

    // Value of a scalar option
    auto value(const std::string& key) const -> const fortran_utils::NamelistValue& {
        return m_options.at(key).values.front();
    }

    auto values(const std::string& key) const
        -> const std::vector<fortran_utils::NamelistValue>& {
        return m_options.at(key).values;
    }

    // Copy of the config, with the value of an existing scalar option replaced
    auto withValue(const std::string& key, fortran_utils::NamelistValue value) const
        -> ConfigDict {
        return withValues(key, {std::move(value)});
    }

    // Copy of the config, with the values of an existing option replaced
    auto withValues(const std::string& key, std::vector<fortran_utils::NamelistValue> values)
        const -> ConfigDict;

    auto getInt(const std::string& key) const -> int;

    auto getBool(const std::string& key) const -> bool {
        return std::get<bool>(value(key));
    }

    auto getDouble(const std::string& key) const -> double {
        return std::get<double>(value(key));
    }

    auto getString(const std::string& key) const -> const std::string& {
        return std::get<std::string>(value(key));
    }
};

// Defaults of the general options
auto defaultMlipConfig() -> ConfigDict;

// Defaults of the ML/MM options
auto defaultMlmmConfig() -> ConfigDict;

// Defaults of the options of Amber's &qmmm namelist that are needed for ML/MM. The
// other options of the group are ignored
auto defaultAmberQmmmConfig() -> ConfigDict;

// An invalid combination of config options
class config_validation_error : public std::runtime_error {
    public:
    using std::runtime_error::runtime_error;
};

// A device that libtorchani was not built to support
class unsupported_device_error : public std::runtime_error {
    public:
    using std::runtime_error::runtime_error;
};

/**
 * Holds global TorchANI-Interface device and precision
 * */
class GlobalConfig {
    torch::TensorOptions m_opts{
        torch::TensorOptions().device(torch::Device{torch::kCPU, -1}).dtype(torch::kFloat)
    };
    bool m_use_no_eval_atoms = false;
    bool m_mlip_interface_is_init = false;
    std::optional<ConfigDict> m_mlip_config;
    std::optional<ConfigDict> m_mlmm_config;
    // From Amber's &qmmm namelist, if read
    int m_amber_qmmm_int = 1;

    // Mirrors ani_nml_validate in sander, assuming qmmm_int = 1
    static auto validateConfig(
        const ConfigDict& mlip_config,
        const std::optional<ConfigDict>& mlmm_config
    ) -> void;

    static auto usesInternalOptions(const ConfigDict& mlmm_config) -> bool;

    // Mirrors the checks of sander that depend on its &qmmm namelist
    static auto validateAmberQmmmNamelist(
        const ConfigDict& mlip_config,
        const ConfigDict& mlmm_config,
        const ConfigDict& amber_qmmm_config
    ) -> void;

    // The general or ML/MM config dict that holds the option. Throws if there is none
    auto configDictWithOption(const std::string& key) const -> const ConfigDict&;

    public:
    GlobalConfig() = default;

    explicit GlobalConfig(
        ConfigDict mlip_config,
        std::optional<ConfigDict> mlmm_config = std::nullopt,
        std::optional<ConfigDict> amber_qmmm_config = std::nullopt
    );

    // Read the config from the first "&mlip" namelist in an Amber input file (mdin).
    // If read_mlmm_config, also read the ML/MM options. If calling_program is "sander"
    // too, also read the first "&qmmm" namelist, and validate the config against it.
    // calling_program is "none", "sander" or "pmemd" (case-insensitive)
    static auto fromNamelistFile(
        const std::string& fpath,
        bool read_mlmm_config = false,
        const std::string& calling_program = "none"
    ) -> GlobalConfig;

    auto dtype() const -> torch::ScalarType {
        return m_opts.dtype().toScalarType();
    }

    auto device() const -> torch::Device {
        return m_opts.device();
    }

    auto useNoEvalAtoms() const -> bool {
        return m_use_no_eval_atoms;
    }

    auto setNoEvalAtoms(bool use_no_eval_atoms) -> void {
        m_use_no_eval_atoms = use_no_eval_atoms;
    }

    auto setMlChargeAndSpinMult(int ml_charge, int ml_spin_mult) -> void {
        m_mlip_config = mlip()
                            .withValue("ml_charge", static_cast<long>(ml_charge))
                            .withValue("ml_spin_mult", static_cast<long>(ml_spin_mult));
    }

    auto mlipInterfaceIsInit() const -> bool {
        return m_mlip_interface_is_init;
    }

    auto setMlipInterfaceIsInit(bool mlip_interface_is_init) -> void {
        m_mlip_interface_is_init = mlip_interface_is_init;
    }

    auto mlip() const -> const ConfigDict& {
        if (not m_mlip_config) {
            throw std::logic_error(
                "GlobalConfig::mlip was called before the config was read"
            );
        }
        return *m_mlip_config;
    }

    auto hasMlmmConfig() const -> bool {
        return m_mlmm_config.has_value();
    }

    auto mlmm() const -> const ConfigDict& {
        if (not m_mlmm_config) {
            throw std::logic_error(
                "GlobalConfig::mlmm was called but the ML/MM config was not read"
            );
        }
        return *m_mlmm_config;
    }

    // Value of an option, from the general or the ML/MM config. Throws if the option
    // doesn't exist, or is not of the requested type
    auto getBool(const std::string& key) const -> bool {
        return configDictWithOption(key).getBool(key);
    }

    auto getInt(const std::string& key) const -> int {
        return configDictWithOption(key).getInt(key);
    }

    auto getDouble(const std::string& key) const -> double {
        return configDictWithOption(key).getDouble(key);
    }

    auto getString(const std::string& key) const -> const std::string& {
        return configDictWithOption(key).getString(key);
    }

    // Polarizability (AU) of an element, from the ML/MM config. Throws if the ML/MM
    // config was not read, or if the element has no polarizability
    auto getPolarizability(const std::string& element_symbol) const -> double;
    auto getPolarizability(long atomic_num) const -> double;

    // Factor kz (AU) of an element, such that its polarizability is kz times its atomic
    // volume, from the ML/MM config, or a builtin default if volume_kzs is not set.
    // Throws if the ML/MM config was not read, or if the element has no kz
    auto getVolumeKz(long atomic_num) const -> double;

    // Print the config to stdout, and the ML/MM setup it selects, if the ML/MM config
    // was read
    auto printInfo() const -> void;

    auto tensorToTorchaniDevice(torch::Tensor x) const -> torch::Tensor {
        return x.to(m_opts.device());
    }

    auto tensorToTorchaniDtypeAndDevice(torch::Tensor x) const -> torch::Tensor {
        return x.to(m_opts);
    }

    // Amber has PBC enabled in all directions for PME
    auto enabledPbc() const -> torch::Tensor {
        return torch::tensor(
            {true, true, true},
            torch::TensorOptions()
                .dtype(torch::kBool)
                .device(m_opts.device())
        );
    }
};

}  // namespace config
