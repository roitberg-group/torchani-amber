#include "config.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "constants.h"
#include "fortran_utils.h"

namespace config {

namespace {

using fortran_utils::NamelistGroup;
using fortran_utils::NamelistType;

#ifdef TORCHANI_BUILD_CUDA
constexpr bool BUILT_WITH_CUDA = true;
#else
constexpr bool BUILT_WITH_CUDA = false;
#endif

// Null values, and keys not in the group, keep the value in config
ConfigDict updatedConfig(const ConfigDict& config, const NamelistGroup& group) {
    ConfigDict updated_config = config;
    for (const auto& [key, option] : config.options()) {
        auto values_it = group.values.find(key);
        if (values_it == group.values.end()) {
            continue;
        }
        const fortran_utils::NamelistValue& value = values_it->second.front();
        if (!std::holds_alternative<std::monostate>(value)) {
            updated_config = updated_config.withValue(key, value);
        }
    }
    return updated_config;
}

bool isDefault(const ConfigDict& config, const ConfigDict& defaults, const std::string& key) {
    return config.value(key) == defaults.value(key);
}

// Values written as Fortran does, e.g. logicals as T and F
std::string valueString(const fortran_utils::NamelistValue& value) {
    if (std::holds_alternative<bool>(value)) {
        return std::get<bool>(value) ? "T" : "F";
    }
    if (std::holds_alternative<long>(value)) {
        return std::to_string(std::get<long>(value));
    }
    if (std::holds_alternative<double>(value)) {
        std::ostringstream stream;
        stream << std::get<double>(value);
        return stream.str();
    }
    return std::get<std::string>(value);
}

std::string configLine(const ConfigDict& config, const std::string& key) {
    std::ostringstream line;
    line << "ML/MM:  " << std::left << std::setw(29) << key << valueString(config.value(key))
         << "\n";
    return line.str();
}

void printWarning(const std::string& msg) {
    std::cerr << "WARNING (libtorchani)\n" << msg << std::endl;
}

bool isOneOf(const std::string& value, const std::vector<std::string>& options) {
    return std::find(options.begin(), options.end(), value) != options.end();
}

// mlmm_coupling may be one of the deprecated integers, or a string in any case
std::string normalizedMlmmCoupling(const fortran_utils::NamelistValue& value) {
    if (std::holds_alternative<std::string>(value)) {
        std::string mlmm_coupling = std::get<std::string>(value);
        std::transform(
            mlmm_coupling.begin(), mlmm_coupling.end(), mlmm_coupling.begin(), ::tolower
        );
        return mlmm_coupling;
    }
    const std::vector<std::string> names{
        "coulombic", "simple-polarizable", "custom", "none"
    };
    long idx = std::get<long>(value);
    if (idx < 0 || idx >= static_cast<long>(names.size())) {
        throw config_validation_error(
            "Valid integer mlmm_coupling values are 0, 1, 2 and 3 (deprecated)"
        );
    }
    printWarning(
        "mlmm_coupling = " + std::to_string(idx) + " is deprecated, use mlmm_coupling = '"
        + names[idx] + "' instead"
    );
    return names[idx];
}

}  // namespace

auto defaultMlipConfig() -> ConfigDict {
    return ConfigDict{{
        {"model_type", {std::string{"ani2x"}, NamelistType::STR}},
        {"no_eval_mask", {std::string{""}, NamelistType::STR}},
        {"extra_features_file_path", {std::string{""}, NamelistType::STR}},
        {"ml_charge", {0L, NamelistType::INT}},
        {"ml_spin_mult", {1L, NamelistType::INT}},
        {"model_index", {-1L, NamelistType::INT}},
        {"cuda_device_index", {0L, NamelistType::INT}},
        {"use_cuda_device", {BUILT_WITH_CUDA, NamelistType::BOOL}},
        {"use_double_precision", {true, NamelistType::BOOL}},
        {"use_external_neighborlist", {false, NamelistType::BOOL}},
        {"use_external_nonbond", {false, NamelistType::BOOL}},
        {"use_cuaev", {true, NamelistType::BOOL}},
    }};
}

auto defaultMlmmConfig() -> ConfigDict {
    std::map<std::string, ConfigOption> options{
        {"mlmm_coupling", {std::string{"coulombic"}, NamelistType::INT_OR_STR}},
        {"use_mlip_atomic_charges", {false, NamelistType::BOOL}},
        // No longer supported, validation throws if it is False
        {"use_charges_derivatives", {true, NamelistType::BOOL}},
        // No longer available, validation throws if it is True
        {"use_numerical_mlmm_forces", {false, NamelistType::BOOL}},
        // Debug option, should always be False
        {"allow_untested_protocols", {false, NamelistType::BOOL}},
        {"inv_pol_dielectric", {0.5, NamelistType::FLOAT}},
        {"write_ml_coords", {false, NamelistType::BOOL}},
        {"write_ml_forces", {false, NamelistType::BOOL}},
        {"write_ml_atomic_charges", {false, NamelistType::BOOL}},
        // Experimental Hybrid ML-QM potential energy, using a switching function
        {"use_hybrid_mlqm", {false, NamelistType::BOOL}},
        {"hybrid_mlqm_method", {std::string{"none"}, NamelistType::STR}},
        {"hybrid_mlqm_switch_low", {0.2, NamelistType::FLOAT}},
        {"hybrid_mlqm_switch_high", {0.3, NamelistType::FLOAT}},
        // Experimental QM coupling (e.g. from DFTB, ORCA, etc)
        {"use_qm_coupling", {false, NamelistType::BOOL}},
        {"qm_coupling_method", {std::string{"none"}, NamelistType::STR}},
    };
    // Polarizabilities pol_<symbol>. Elements without a default are NaN
    // H taken from Litman, J. M., Liu, C., and Ren, P. (2021).
    // Journal of Chemical Information and Modeling, 62(1), 79-87.
    // Corresponds to the average of all hydrogen types
    // Rest taken from Schwerdtfeger, P., and Nagle, J. K. (2019). Molecular Physics,
    // 117(9-12), 1200-1225.
    const std::array<double, 18> known_polarizabilities{
        3.08, 1.38375, 164.1125, 37.74, 20.5, 11.3, 7.4, 5.3, 3.74,
        2.6611, 162.7, 71.2, 57.8, 37.3, 25.0, 19.4, 14.6, 11.083,
    };
    for (std::size_t i = 0; i < constants::NUM_ELEMENTS; ++i) {
        double polarizability = std::numeric_limits<double>::quiet_NaN();
        if (i < known_polarizabilities.size()) {
            polarizability = known_polarizabilities[i];
        }
        // Namelist keys are lowercased by the parser
        std::string key = "pol_" + std::string(constants::ELEMENT_SYMBOLS[i]);
        std::transform(key.begin(), key.end(), key.begin(), ::tolower);
        options.emplace(key, ConfigOption{polarizability, NamelistType::FLOAT});
    }
    return ConfigDict{options};
}

auto defaultAmberQmmmConfig() -> ConfigDict {
    return ConfigDict{{
        {"qmmm_int", {1L, NamelistType::INT}},
        {"qm_ewald", {1L, NamelistType::INT}},
        {"qmmask", {std::string{""}, NamelistType::STR}},
        {"qmcharge", {0L, NamelistType::INT}},
        {"spin", {1L, NamelistType::INT}},
    }};
}

auto ConfigDict::withValue(const std::string& key, fortran_utils::NamelistValue value) const
    -> ConfigDict {
    std::map<std::string, ConfigOption> options = m_options;
    options.at(key).value = std::move(value);
    return ConfigDict{options};
}

auto ConfigDict::getInt(const std::string& key) const -> int {
    long value = std::get<long>(this->value(key));
    if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) {
        throw std::out_of_range(
            "Value " + std::to_string(value) + " of key " + key + " is out of range"
        );
    }
    return static_cast<int>(value);
}

GlobalConfig::GlobalConfig(
    ConfigDict mlip_config,
    std::optional<ConfigDict> mlmm_config,
    std::optional<ConfigDict> amber_qmmm_config
):
    m_mlip_config{std::move(mlip_config)},
    m_mlmm_config{std::move(mlmm_config)}
{
    if (amber_qmmm_config) {
        m_amber_qmmm_int = amber_qmmm_config->getInt("qmmm_int");
    }
#ifndef TORCHANI_BUILD_CUDA
    if (getBool("use_cuda_device")) {
        throw unsupported_device_error(
            "A CUDA device was requested, but libtorchani was built\n"
            "without CUDA support (TORCHANI_BUILD_CUDA=OFF).\n"
            "Select CPU, or rebuild with TORCHANI_BUILD_CUDA=ON"
        );
    }
#endif
    torch::DeviceType device_type = getBool("use_cuda_device") ? torch::kCUDA : torch::kCPU;
    auto device_idx = static_cast<torch::DeviceIndex>(getInt("cuda_device_index"));
    torch::ScalarType dtype = getBool("use_double_precision") ? torch::kFloat64 : torch::kFloat32;
    m_opts = torch::TensorOptions().device(torch::Device{device_type, device_idx}).dtype(dtype);
}

auto GlobalConfig::fromNamelistFile(
    const std::string& fpath,
    bool read_mlmm_config,
    const std::string& calling_program
) -> GlobalConfig {
    std::string program = calling_program;
    std::transform(program.begin(), program.end(), program.begin(), ::tolower);
    if (!isOneOf(program, {"none", "sander", "pmemd"})) {
        throw std::runtime_error(
            "calling_program must be 'none', 'sander' or 'pmemd', but it is '"
            + calling_program + "'"
        );
    }
    // Relative paths are resolved wrt the current working directory
    std::string abs_fpath = std::filesystem::absolute(fpath).string();
    std::ifstream file(abs_fpath);
    if (!file) {
        throw std::runtime_error("Could not open " + abs_fpath);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string text = buffer.str();

    std::vector<std::string> group_names = fortran_utils::listNamelistGroupNames(text);
    bool file_has_mlip = std::count(group_names.begin(), group_names.end(), "mlip") > 0;
    bool file_has_ani = std::count(group_names.begin(), group_names.end(), "ani") > 0;
    if (file_has_mlip && file_has_ani) {
        throw std::runtime_error(
            "Both &mlip and &ani namelists found in " + abs_fpath + ", only one can be used"
        );
    }
    if (!file_has_mlip && !file_has_ani) {
        throw std::runtime_error("No &mlip namelist found in " + abs_fpath);
    }
    if (file_has_ani) {
        std::cerr << "WARNING (libtorchani)\n"
                  << "The &ani namelist in " << abs_fpath << " is deprecated,"
                  << " use &mlip instead" << std::endl;
    }
    std::string group_name = file_has_mlip ? "mlip" : "ani";

    ConfigDict mlip_config = defaultMlipConfig();
    std::optional<ConfigDict> mlmm_config;
    if (read_mlmm_config) {
        mlmm_config = defaultMlmmConfig();
    }
    fortran_utils::NamelistSchema schema;
    for (const auto& [key, option] : mlip_config.options()) {
        schema.emplace(key, option.type);
    }
    if (mlmm_config) {
        for (const auto& [key, option] : mlmm_config->options()) {
            schema.emplace(key, option.type);
        }
        // Deprecated no-ops, only checked below
        schema.emplace("use_torch_coupling", NamelistType::BOOL);
        schema.emplace("write_charges_grad", NamelistType::BOOL);
    }

    // List of pairs {<deprecated>, <canonical>}
    const fortran_utils::NamelistAliases aliases{
        {"net_charge", "ml_charge"},
        {"use_amber_neighborlist", "use_external_neighborlist"},
        {"use_all_amber_nonbond", "use_external_nonbond"},
        {"use_torchani_charges", "use_mlip_atomic_charges"},
        {"write_xyz", "write_ml_coords"},
        {"write_forces", "write_ml_forces"},
        {"write_charges", "write_ml_atomic_charges"},
        {"use_numerical_qmmm_forces", "use_numerical_mlmm_forces"},
        {"use_switching_function", "use_hybrid_mlqm"},
        {"switching_program", "hybrid_mlqm_method"},
        {"qlow", "hybrid_mlqm_switch_low"},
        {"qhigh", "hybrid_mlqm_switch_high"},
        {"use_extcoupling", "use_qm_coupling"},
        {"extcoupling_program", "qm_coupling_method"},
    };
    std::vector<NamelistGroup> groups =
        fortran_utils::parseNamelists(text, group_name, schema, aliases);
    const NamelistGroup& group = groups.front();

    mlip_config = updatedConfig(mlip_config, group);
    if (mlmm_config) {
        mlmm_config = updatedConfig(*mlmm_config, group);
        mlmm_config = mlmm_config->withValue(
            "mlmm_coupling", normalizedMlmmCoupling(mlmm_config->value("mlmm_coupling"))
        );
        if (group.values.count("use_torch_coupling") > 0) {
            printWarning(
                "use_torch_coupling is deprecated and has no effect, the ML/MM coupling is"
                " always calculated by the MLIP Interface"
            );
        }
        if (group.values.count("write_charges_grad") > 0) {
            printWarning(
                "write_charges_grad is deprecated and has no effect, the gradients of the"
                " charges are no longer written"
            );
        }
    }
    validateConfig(mlip_config, mlmm_config);

    std::optional<ConfigDict> amber_qmmm_config;
    // Only sander supports ML/MM, through its &qmmm namelist
    if (read_mlmm_config && program == "sander") {
        if (std::count(group_names.begin(), group_names.end(), "qmmm") == 0) {
            throw std::runtime_error("No &qmmm namelist found in " + abs_fpath);
        }
        ConfigDict qmmm_defaults = defaultAmberQmmmConfig();
        fortran_utils::NamelistSchema qmmm_schema;
        for (const auto& [key, option] : qmmm_defaults.options()) {
            qmmm_schema.emplace(key, option.type);
        }
        // Only some of the options of the group are read, the others are ignored
        std::vector<NamelistGroup> qmmm_groups = fortran_utils::parseNamelists(
            text, "qmmm", qmmm_schema, {}, /*ignore_unknown_keys=*/true
        );
        const NamelistGroup& qmmm_group = qmmm_groups.front();
        amber_qmmm_config = updatedConfig(qmmm_defaults, qmmm_group);
        // Values set only in &qmmm are used for &mlip, and values set in both must agree
        const std::vector<std::pair<std::string, std::string>> mlip_and_qmmm_keys = {
            {"ml_charge", "qmcharge"},
            {"ml_spin_mult", "spin"},
        };
        for (const auto& [mlip_key, qmmm_key] : mlip_and_qmmm_keys) {
            bool mlip_has_key = group.values.count(mlip_key) > 0;
            bool qmmm_has_key = qmmm_group.values.count(qmmm_key) > 0;
            // sander uses qmcharge before the MLIP Interface is initialized (adjust_q), so
            // it can't be overriden with ml_charge
            if (mlip_key == "ml_charge" and mlip_has_key and not qmmm_has_key
                and mlip_config.getInt(mlip_key) != 0) {
                throw config_validation_error(
                    "ml_charge = " + std::to_string(mlip_config.getInt(mlip_key))
                    + " in &mlip requires the same qmcharge in &qmmm"
                );
            }
            if (not qmmm_has_key) {
                continue;
            }
            int qmmm_value = amber_qmmm_config->getInt(qmmm_key);
            if (not mlip_has_key) {
                mlip_config = mlip_config.withValue(mlip_key, static_cast<long>(qmmm_value));
            } else if (mlip_config.getInt(mlip_key) != qmmm_value) {
                throw config_validation_error(
                    mlip_key + " = " + std::to_string(mlip_config.getInt(mlip_key))
                    + " in &mlip doesn't match " + qmmm_key + " = "
                    + std::to_string(qmmm_value) + " in &qmmm"
                );
            }
        }
        validateAmberQmmmNamelist(mlip_config, *mlmm_config, *amber_qmmm_config);
    }
    return GlobalConfig{mlip_config, mlmm_config, amber_qmmm_config};
}

auto GlobalConfig::validateConfig(
    const ConfigDict& mlip_config,
    const std::optional<ConfigDict>& mlmm_config
) -> void {
    if (mlip_config.getString("model_type").find("::") != std::string::npos) {
        throw config_validation_error(
            "model_type can't contain '::', use extra_features_file_path to specify the"
            " extra features file"
        );
    }
    if (!mlmm_config) {
        return;
    }
    const ConfigDict& mlmm = *mlmm_config;
    const ConfigDict defaults = defaultMlmmConfig();
    const std::string& mlmm_coupling = mlmm.getString("mlmm_coupling");
    bool use_mlip_atomic_charges = mlmm.getBool("use_mlip_atomic_charges");
    bool use_hybrid_mlqm = mlmm.getBool("use_hybrid_mlqm");
    bool use_qm_coupling = mlmm.getBool("use_qm_coupling");
    const std::string& hybrid_mlqm_method = mlmm.getString("hybrid_mlqm_method");
    const std::string& qm_coupling_method = mlmm.getString("qm_coupling_method");
    bool inv_pol_dielectric_is_default = isDefault(mlmm, defaults, "inv_pol_dielectric");

    if (!isOneOf(mlmm_coupling, {"coulombic", "simple-polarizable", "custom", "none"})) {
        throw config_validation_error(
            "Valid mlmm_coupling values are 'coulombic', 'simple-polarizable', 'custom' and"
            " 'none'"
        );
    }
    if (!mlmm.getBool("use_charges_derivatives")) {
        throw config_validation_error(
            "use_charges_derivatives = F is no longer supported, the derivatives of the"
            " charges are always taken into account"
        );
    }
    if (mlmm.getBool("use_numerical_mlmm_forces")) {
        throw config_validation_error(
            "use_numerical_mlmm_forces (formerly use_numerical_qmmm_forces) is no longer"
            " available"
        );
    }
    if (mlip_config.getBool("use_external_neighborlist")) {
        throw config_validation_error(
            "use_external_neighborlist is not supported for ML/MM simulations"
        );
    }
    if (use_hybrid_mlqm && use_qm_coupling) {
        throw config_validation_error(
            "use_hybrid_mlqm and use_qm_coupling can't be used together"
        );
    }
    if (use_qm_coupling && mlmm_coupling != "none") {
        throw config_validation_error(
            "use_qm_coupling = T requires mlmm_coupling = 'none', since the ML/MM coupling"
            " is calculated by the helper QM program"
        );
    }
    if (use_hybrid_mlqm && mlmm_coupling == "custom") {
        throw config_validation_error(
            "use_hybrid_mlqm is not supported with mlmm_coupling = 'custom'"
        );
    }
    if (!mlmm.getBool("allow_untested_protocols")) {
        if (mlmm_coupling == "coulombic" && use_mlip_atomic_charges) {
            throw config_validation_error(
                "mlmm_coupling = 'coulombic' is untested with use_mlip_atomic_charges = T,"
                " set allow_untested_protocols = T if you really mean to run this simulation"
            );
        }
        if (mlmm_coupling == "simple-polarizable" && !use_mlip_atomic_charges) {
            throw config_validation_error(
                "mlmm_coupling = 'simple-polarizable' is untested with fixed topology"
                " charges, set allow_untested_protocols = T if you really mean to run this"
                " simulation"
            );
        }
    }
    if (!inv_pol_dielectric_is_default && mlmm_coupling != "simple-polarizable") {
        throw config_validation_error(
            "inv_pol_dielectric only has an effect with mlmm_coupling = 'simple-polarizable'"
        );
    }
    if (mlmm_coupling == "custom") {
        if (use_mlip_atomic_charges) {
            throw config_validation_error(
                "mlmm_coupling = 'custom' handles charges internally,"
                " set use_mlip_atomic_charges = F"
            );
        }
        if (mlmm.getBool("write_ml_atomic_charges")) {
            throw config_validation_error(
                "mlmm_coupling = 'custom' returns no charges, set write_ml_atomic_charges = F"
            );
        }
    }
    if (use_qm_coupling && !isOneOf(qm_coupling_method, {"amber-dftb", "orca", "gau", "lio"})) {
        throw config_validation_error(
            "Valid qm_coupling_method values are 'amber-dftb', 'orca', 'gau' and 'lio'"
        );
    }
    if (!use_qm_coupling && !isDefault(mlmm, defaults, "qm_coupling_method")) {
        throw config_validation_error(
            "qm_coupling_method has no effect with use_qm_coupling = F"
        );
    }
    if (!use_hybrid_mlqm && !isDefault(mlmm, defaults, "hybrid_mlqm_method")) {
        throw config_validation_error(
            "hybrid_mlqm_method has no effect with use_hybrid_mlqm = F"
        );
    }
    if (use_hybrid_mlqm && !isOneOf(hybrid_mlqm_method, {"orca", "gau", "lio"})) {
        throw config_validation_error(
            "Valid hybrid_mlqm_method values are 'orca', 'gau' and 'lio'"
        );
    }

    if (usesInternalOptions(mlmm)) {
        printWarning(
            "You are using experimental or deprecated options, which will break in future"
            " releases. Results may be wrong, use at your own risk"
        );
    }
    if (!inv_pol_dielectric_is_default) {
        printWarning(
            "The recommended value of inv_pol_dielectric is 0.5, but "
            + std::to_string(mlmm.getDouble("inv_pol_dielectric")) + " was specified"
        );
    }
    if (use_qm_coupling && use_mlip_atomic_charges) {
        printWarning(
            "use_mlip_atomic_charges has no effect with use_qm_coupling = T, the QM/MM"
            " coupling is calculated by the helper QM program"
        );
    }
    if (!use_qm_coupling) {
        if (mlmm_coupling == "coulombic" && use_mlip_atomic_charges) {
            printWarning(
                "You are using an untested protocol: mlmm_coupling = 'coulombic' is only"
                " tested with fixed topology charges"
            );
        }
        if (mlmm_coupling == "simple-polarizable" && !use_mlip_atomic_charges) {
            printWarning(
                "You are using an untested protocol: mlmm_coupling = 'simple-polarizable'"
                " is only tested with use_mlip_atomic_charges = T"
            );
        }
    }
}

auto GlobalConfig::validateAmberQmmmNamelist(
    [[maybe_unused]] const ConfigDict& mlip_config,
    const ConfigDict& mlmm_config,
    const ConfigDict& amber_qmmm_config
) -> void {
    int qmmm_int = amber_qmmm_config.getInt("qmmm_int");
    const std::string& mlmm_coupling = mlmm_config.getString("mlmm_coupling");

    if (qmmm_int != 0 && qmmm_int != 1 && qmmm_int != 5) {
        throw config_validation_error(
            "ML/MM requires qmmm_int = 1 (0 and 5 are internal options for debugging only)"
        );
    }
    if (qmmm_int != 1 && !mlmm_config.getBool("allow_untested_protocols")) {
        throw config_validation_error(
            "qmmm_int = 1 is the only supported option for ML/MM, other options are untested,"
            " set allow_untested_protocols = T if you really mean to run this simulation"
        );
    }
    if (qmmm_int == 0 && mlmm_coupling != "none") {
        throw config_validation_error("qmmm_int = 0 requires mlmm_coupling = 'none'");
    }
    if (qmmm_int == 5 && mlmm_coupling != "none") {
        throw config_validation_error("qmmm_int = 5 requires mlmm_coupling = 'none'");
    }
    if (qmmm_int != 1 && mlmm_config.getBool("use_qm_coupling")) {
        throw config_validation_error("use_qm_coupling requires qmmm_int = 1");
    }

    if (qmmm_int == 0) {
        printWarning(
            "You are using an untested protocol: qmmm_int = 0 turns off the QM/MM"
            " electrostatic coupling. You probably want qmmm_int = 1"
        );
    }
    if (qmmm_int == 5) {
        printWarning(
            "You are using an untested protocol: with qmmm_int = 5 the QM/MM electrostatic"
            " coupling is calculated by sander, with the mechanical embedding approximation"
            " and fixed QM-region charges read from the topology file. You probably want"
            " qmmm_int = 1"
        );
    }
}

auto GlobalConfig::usesInternalOptions(const ConfigDict& mlmm_config) -> bool {
    const ConfigDict defaults = defaultMlmmConfig();
    return mlmm_config.getBool("use_hybrid_mlqm")
        || !isDefault(mlmm_config, defaults, "hybrid_mlqm_method")
        || !isDefault(mlmm_config, defaults, "hybrid_mlqm_switch_low")
        || !isDefault(mlmm_config, defaults, "hybrid_mlqm_switch_high")
        || mlmm_config.getBool("use_qm_coupling")
        || !isDefault(mlmm_config, defaults, "qm_coupling_method");
}

auto GlobalConfig::configDictWithOption(const std::string& key) const
    -> const ConfigDict& {
    if (mlip().hasOption(key)) {
        return mlip();
    }
    if (mlmm().hasOption(key)) {
        return mlmm();
    }
    throw std::out_of_range("Unknown config option " + key);
}

auto GlobalConfig::getPolarizability(const std::string& element_symbol) const -> double {
    std::string key = "pol_" + element_symbol;
    std::transform(key.begin(), key.end(), key.begin(), ::tolower);
    double polarizability = getDouble(key);
    if (std::isnan(polarizability)) {
        throw std::runtime_error(
            "The polarizability " + key
            + " has no default, it must be specified in the namelist"
        );
    }
    return polarizability;
}

auto GlobalConfig::getPolarizability(long atomic_num) const -> double {
    if (atomic_num < 1 || atomic_num > static_cast<long>(constants::NUM_ELEMENTS)) {
        throw std::out_of_range("Invalid atomic number " + std::to_string(atomic_num));
    }
    return getPolarizability(std::string(constants::ELEMENT_SYMBOLS[atomic_num - 1]));
}

auto GlobalConfig::printInfo() const -> void {
    std::ostringstream out;
    out << "ML/MM:------------------ML/MM config----------------\n";
    for (const char* key : {
             "model_type",
             "model_index",
             "use_cuda_device",
             "use_double_precision",
             "use_cuaev",
             "cuda_device_index",
         }) {
        out << configLine(mlip(), key);
    }
    if (!getString("extra_features_file_path").empty()) {
        out << configLine(mlip(), "extra_features_file_path");
    }
    if (!m_mlmm_config) {
        out << "ML/MM:--------------end config-------------------\n";
        std::cout << out.str() << std::flush;
        return;
    }
    const std::string& mlmm_coupling = getString("mlmm_coupling");
    bool use_qm_coupling = getBool("use_qm_coupling");
    const std::string& qm_coupling_method = getString("qm_coupling_method");
    for (const char* key : {
             "write_ml_coords",
             "write_ml_forces",
             "write_ml_atomic_charges",
             "use_mlip_atomic_charges",
         }) {
        out << configLine(mlmm(), key);
    }
    if (m_amber_qmmm_int == 1) {
        out << configLine(mlmm(), "mlmm_coupling");
    }
    if (usesInternalOptions(mlmm())) {
        out << "ML/MM:------WARNING! INTERNAL OPTIONS, DO NOT USE------\n";
        for (const char* key : {
                 "hybrid_mlqm_method",
                 "use_hybrid_mlqm",
                 "hybrid_mlqm_switch_low",
                 "hybrid_mlqm_switch_high",
                 "inv_pol_dielectric",
                 "use_qm_coupling",
                 "qm_coupling_method",
             }) {
            out << configLine(mlmm(), key);
        }
    }
    out << "ML/MM:--------------end config-------------------\n";

    if (use_qm_coupling) {
        out << "ML/MM: This is an EXPERIMENTAL feature, don't use in production code!\n"
            << "     QM coupling functionality of MLIP Interface is ON\n"
            << "     The selected method for QM coupling is: " << qm_coupling_method << "\n";
    }
    if (getBool("use_hybrid_mlqm")) {
        out << "ML/MM: This is an EXPERIMENTAL feature, don't use in production code!\n"
            << "     Hybrid ML-QM functionality of MLIP Interface is ON\n"
            << "     The selected hybrid ML-QM method is: "
            << getString("hybrid_mlqm_method") << "\n";
    }
    out << "ML/MM ELECTROSTATIC COUPLING AND CHARGE CONFIG:\n";
    if (m_amber_qmmm_int != 1) {
        std::cout << out.str() << std::flush;
        return;
    }
    if (mlmm_coupling == "none" && !use_qm_coupling) {
        out << "ML/MM: Will use no QM/MM interaction energy, only the in vacuo ML energy\n";
        std::cout << out.str() << std::flush;
        return;
    }
    out << "ML/MM: QM/MM electrostatic coupling will be calculated by MLIP Interface\n";
    if (use_qm_coupling) {
        out << "ML/MM: This is an EXPERIMENTAL feature, don't use in production code!\n"
            << "     Will delegate QM/MM electrostatic coupling to " << qm_coupling_method << "\n"
            << "     The electrostatic coupling approximation will be used\n"
            << "     The QM energies and forces will be also corrected\n";
        std::cout << out.str() << std::flush;
        return;
    }
    if (mlmm_coupling == "coulombic") {
        out << "ML/MM: Will use a coulombic-only QM/MM interaction energy,"
            << " with no polarization corrections\n";
    } else if (mlmm_coupling == "simple-polarizable") {
        out << "ML/MM: Will use the simple polarizable approximation for the QM/MM"
            << " interaction energy\n"
            << "     Thole-inspired correction terms will be added to the coulombic"
            << " interaction\n"
            << "     This partially accounts for QM-region electrostatic polarization and"
            << " distortion\n"
            << "     If you use this setting please cite"
            << " 'https://doi.org/10.1021/acs.jcim.4c00478'\n";
    } else {
        out << "ML/MM: Will use a fully custom ML/MM model (mlmm_coupling = 'custom')\n"
            << "     The model receives ML species+coords and MM coords+charges\n"
            << "     and computes the coupling internally.\n"
            << "     Only models exposing custom_mlmm() support this kind of coupling.\n";
    }
    if (mlmm_coupling != "custom" && getBool("use_mlip_atomic_charges")) {
        out << "ML/MM: Will use position-dependent NN-predicted charges for the QM-region,"
            << " taking into account dq/dr\n";
    } else if (mlmm_coupling != "custom") {
        out << "ML/MM: Will use fixed charges read from the topology file for the QM-region\n";
    }
    std::cout << out.str() << std::flush;
}

}  // namespace config
