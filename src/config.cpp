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

// Values are assigned element-wise. Null values, and keys not in the group, keep the
// values in config
ConfigDict updatedConfig(const ConfigDict& config, const NamelistGroup& group) {
    ConfigDict updated_config = config;
    for (const auto& [key, option] : config.options()) {
        auto values_it = group.values.find(key);
        if (values_it == group.values.end()) {
            continue;
        }
        const std::vector<fortran_utils::NamelistValue>& new_values = values_it->second;
        std::vector<fortran_utils::NamelistValue> values = option.values;
        if (values.size() < new_values.size()) {
            values.resize(new_values.size());
        }
        for (std::size_t i = 0; i != new_values.size(); ++i) {
            if (!std::holds_alternative<std::monostate>(new_values[i])) {
                values[i] = new_values[i];
            }
        }
        updated_config = updated_config.withValues(key, values);
    }
    return updated_config;
}

bool isDefault(const ConfigDict& config, const ConfigDict& defaults, const std::string& key) {
    return config.values(key) == defaults.values(key);
}

// Values written as Fortran does, e.g. logicals as T and F, and nulls as empty strings
std::string valueString(const fortran_utils::NamelistValue& value) {
    if (std::holds_alternative<std::monostate>(value)) {
        return "";
    }
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
    line << "ML/MM:  " << std::left << std::setw(29) << key;
    const std::vector<fortran_utils::NamelistValue>& values = config.values(key);
    for (std::size_t i = 0; i != values.size(); ++i) {
        if (i != 0) {
            line << ", ";
        }
        line << valueString(values[i]);
    }
    line << "\n";
    return line.str();
}

void printWarning(const std::string& msg) {
    std::cerr << "WARNING (libtorchani)\n" << msg << std::endl;
}

bool isOneOf(const std::string& value, const std::vector<std::string>& options) {
    return std::find(options.begin(), options.end(), value) != options.end();
}

// volume_kzs may be the name of one of these, or the kz of each element. They are
// indexed by atomic number, starting at H. NaN means the element has no kz
constexpr double NO_KZ = std::numeric_limits<double>::quiet_NaN();
const std::map<std::string, std::vector<double>> BUILTIN_VOLUME_KZS{
    {
        "builtin-mbis-charges",
        {
            0.0, NO_KZ, NO_KZ, NO_KZ, NO_KZ, 0.7190062581351027, 0.9894465754783003,
            0.5712807132434742, NO_KZ, NO_KZ, NO_KZ, NO_KZ, NO_KZ, NO_KZ, NO_KZ, 0.0,
            0.47491014980647467,
        },
    },
    {
        "builtin-fixed-resp-charges",
        {
            0.7659661617582175, NO_KZ, NO_KZ, NO_KZ, NO_KZ, 0.36055843774225754,
            0.32324322791847215, 0.26476340834055967, NO_KZ, NO_KZ, NO_KZ, NO_KZ, NO_KZ,
            NO_KZ, NO_KZ, 0.030818228617190933, 1.1044386233759735,
        },
    },
};

// Builtin volume_kzs used if none are given
std::string defaultVolumeKzsName(const ConfigDict& mlmm) {
    if (mlmm.getBool("use_mlip_atomic_charges")) {
        return "builtin-mbis-charges";
    }
    return "builtin-fixed-resp-charges";
}

// The kz of each element, indexed by atomic number starting at H, NaN if the element has
// no kz. Builtin names are replaced by their values
std::vector<double> resolvedVolumeKzs(const ConfigDict& mlmm) {
    std::vector<fortran_utils::NamelistValue> volume_kzs = mlmm.values("volume_kzs");
    if (volume_kzs.empty()) {
        volume_kzs = {defaultVolumeKzsName(mlmm)};
    }
    std::vector<double> kzs(constants::NUM_ELEMENTS, NO_KZ);
    if (volume_kzs.size() == 1 && std::holds_alternative<std::string>(volume_kzs.front())) {
        const std::string& name = std::get<std::string>(volume_kzs.front());
        auto builtin_it = BUILTIN_VOLUME_KZS.find(name);
        if (builtin_it == BUILTIN_VOLUME_KZS.end()) {
            throw config_validation_error(
                "Valid volume_kzs names are 'builtin-mbis-charges' and"
                " 'builtin-fixed-resp-charges'"
            );
        }
        std::copy(builtin_it->second.begin(), builtin_it->second.end(), kzs.begin());
        return kzs;
    }
    if (volume_kzs.size() > constants::NUM_ELEMENTS) {
        throw config_validation_error(
            "volume_kzs has more values than there are elements"
        );
    }
    for (std::size_t i = 0; i != volume_kzs.size(); ++i) {
        if (std::holds_alternative<std::string>(volume_kzs[i])) {
            throw config_validation_error(
                "volume_kzs must be a single builtin name, or a list of numbers"
            );
        }
        if (std::holds_alternative<double>(volume_kzs[i])) {
            kzs[i] = std::get<double>(volume_kzs[i]);
        }
    }
    return kzs;
}

// GEEP widths (Angstrom) used by pmemd, for H to Xe, indexed by atomic number starting
// at H. Elements past Xe have no width
const std::vector<double> PMEMD_GEEP_SIGMAS_ANG{
    1.00, 1.20, 1.60, 1.50, 1.50, 1.50, 1.40, 1.30, 1.25, 1.30,  // H to Ne
    1.80, 1.70, 1.70, 1.65, 1.60, 1.60, 1.55, 1.50, 2.00, 1.90,  // Na to Ca
    1.85, 1.80, 1.80, 1.75, 1.75, 1.75, 1.70, 1.70, 1.70, 1.70,  // Sc to Zn
    1.75, 1.70, 1.70, 1.70, 1.70, 1.70, 2.10, 2.00, 1.95, 1.90,  // Ga to Zr
    1.90, 1.90, 1.85, 1.85, 1.85, 1.85, 1.85, 1.85, 1.90, 1.90,  // Nb to Sn
    1.90, 1.90, 1.85, 1.80,  // Sb to Xe
};

// geep_sigmas may be "builtin" (the pmemd widths), "builtin-carbon" (the pmemd width of
// carbon for all elements up to Xe) or the width of each element, in bohr
std::vector<double> builtinGeepSigmas(const std::string& name) {
    constexpr std::size_t carbon_idx = 5;
    std::vector<double> sigmas = PMEMD_GEEP_SIGMAS_ANG;
    if (name == "builtin-carbon") {
        std::fill(sigmas.begin(), sigmas.end(), PMEMD_GEEP_SIGMAS_ANG[carbon_idx]);
    } else if (name != "builtin") {
        throw config_validation_error(
            "Valid geep_sigmas names are 'builtin' and 'builtin-carbon'"
        );
    }
    for (double& sigma : sigmas) {
        sigma *= constants::CODATA08_ANGSTROM_TO_BOHR;
    }
    return sigmas;
}

// The width (bohr) of each element, indexed by atomic number starting at H, NaN if the
// element has no width. Builtin names are replaced by their values. geep_factor is not
// applied
std::vector<double> resolvedGeepSigmas(const ConfigDict& mlmm) {
    std::vector<fortran_utils::NamelistValue> geep_sigmas = mlmm.values("geep_sigmas");
    if (geep_sigmas.empty()) {
        geep_sigmas = {std::string{"builtin"}};
    }
    std::vector<double> sigmas(
        constants::NUM_ELEMENTS, std::numeric_limits<double>::quiet_NaN()
    );
    if (geep_sigmas.size() == 1 && std::holds_alternative<std::string>(geep_sigmas.front())) {
        std::vector<double> builtin_sigmas =
            builtinGeepSigmas(std::get<std::string>(geep_sigmas.front()));
        std::copy(builtin_sigmas.begin(), builtin_sigmas.end(), sigmas.begin());
        return sigmas;
    }
    if (geep_sigmas.size() > constants::NUM_ELEMENTS) {
        throw config_validation_error(
            "geep_sigmas has more values than there are elements"
        );
    }
    for (std::size_t i = 0; i != geep_sigmas.size(); ++i) {
        if (std::holds_alternative<std::string>(geep_sigmas[i])) {
            throw config_validation_error(
                "geep_sigmas must be a single builtin name, or a list of numbers"
            );
        }
        if (std::holds_alternative<double>(geep_sigmas[i])) {
            sigmas[i] = std::get<double>(geep_sigmas[i]);
            if (sigmas[i] <= 0.0) {
                throw config_validation_error("geep_sigmas must be > 0");
            }
        }
    }
    return sigmas;
}

// With mechanical embedding the MD engine calculates the ML/MM coupling
bool isMechanicalEmbedding(const std::string& mlmm_coupling) {
    return mlmm_coupling == "mechanical-embedding"
        || mlmm_coupling == "mechanical-embedding-no-electrostatics";
}

// mlmm_coupling may be one of the deprecated integers, or a string in any case. 'me' and
// 'me-no-electrostatics' are aliases of 'mechanical-embedding' and
// 'mechanical-embedding-no-electrostatics'
std::string normalizedMlmmCoupling(const fortran_utils::NamelistValue& value) {
    if (std::holds_alternative<std::string>(value)) {
        std::string mlmm_coupling = std::get<std::string>(value);
        std::transform(
            mlmm_coupling.begin(), mlmm_coupling.end(), mlmm_coupling.begin(), ::tolower
        );
        if (mlmm_coupling == "me") {
            return "mechanical-embedding";
        }
        if (mlmm_coupling == "me-no-electrostatics") {
            return "mechanical-embedding-no-electrostatics";
        }
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

auto isPolarizableCoupling(const std::string& mlmm_coupling) -> bool {
    return mlmm_coupling == "simple-polarizable"
        || mlmm_coupling == "simple-polarizable-geep";
}

auto isGeepCoupling(const std::string& mlmm_coupling) -> bool {
    return mlmm_coupling == "coulombic-geep" || mlmm_coupling == "simple-polarizable-geep";
}

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
        {"write_coords", {false, NamelistType::BOOL}},
        {"write_forces", {false, NamelistType::BOOL}},
        {"write_atomic_charges", {false, NamelistType::BOOL}},
        {"write_atomic_volumes", {false, NamelistType::BOOL}},
        {"write_atomic_alphas", {false, NamelistType::BOOL}},
        // The write_* outputs are written at MD steps that are multiples of write_interval
        {"write_interval", {1L, NamelistType::INT}},
        // Dir of the write_* output files, the current working directory if empty
        {"output_dir", {std::string{""}, NamelistType::STR}},
    }};
}

auto defaultMlmmConfig() -> ConfigDict {
    std::map<std::string, ConfigOption> options{
        {"mlmm_coupling", {std::string{"coulombic"}, NamelistType::INT_OR_STR}},
        {"use_mlip_atomic_charges", {false, NamelistType::BOOL}},
        {"use_mlip_atomic_volumes", {false, NamelistType::BOOL}},
        {"volume_kzs", {std::vector<fortran_utils::NamelistValue>{}, NamelistType::VEC_FLOAT_OR_STR}},
        // 'builtin' if empty
        {"geep_sigmas", {std::vector<fortran_utils::NamelistValue>{}, NamelistType::VEC_FLOAT_OR_STR}},
        {"geep_factor", {1.0, NamelistType::FLOAT}},
        // No longer supported, validation throws if it is False
        {"use_charges_derivatives", {true, NamelistType::BOOL}},
        // No longer available, validation throws if it is True
        {"use_numerical_mlmm_forces", {false, NamelistType::BOOL}},
        // Debug option, should always be False
        {"allow_untested_protocols", {false, NamelistType::BOOL}},
        {"inv_pol_dielectric", {0.5, NamelistType::FLOAT}},
        // Used by sander as qmcut, qmmask and qmshake, if ifmlmm = 1
        {"mm_cutoff", {15.0, NamelistType::FLOAT}},
        {"ml_mask", {std::string{""}, NamelistType::STR}},
        {"ml_shake", {true, NamelistType::BOOL}},
        // Set from mlmm_coupling if empty, it only supports one value for each coupling
        {"mm_cutoff_kind", {std::string{""}, NamelistType::STR}},
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

auto ConfigDict::withValues(
    const std::string& key, std::vector<fortran_utils::NamelistValue> values
) const -> ConfigDict {
    std::map<std::string, ConfigOption> options = m_options;
    options.at(key).values = std::move(values);
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
    const std::string& from_namelist,
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
    std::string source = from_namelist;
    std::transform(source.begin(), source.end(), source.begin(), ::tolower);
    if (!isOneOf(source, {"mlip", "qmmm-and-mlip", "mlmm"})) {
        throw std::runtime_error(
            "from_namelist must be 'mlip', 'qmmm-and-mlip' or 'mlmm', but it is '"
            + from_namelist + "'"
        );
    }
    bool read_mlmm_config = source != "mlip";
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
    bool file_has_qmmm = std::count(group_names.begin(), group_names.end(), "qmmm") > 0;
    std::string group_name = "mlmm";
    if (source == "mlmm") {
        if (std::count(group_names.begin(), group_names.end(), "mlmm") == 0) {
            throw std::runtime_error("No &mlmm namelist found in " + abs_fpath);
        }
        if (file_has_qmmm) {
            throw config_validation_error(
                "A &qmmm namelist was found in " + abs_fpath
                + ", it can't be used together with &mlmm"
            );
        }
    } else {
        bool file_has_mlip = std::count(group_names.begin(), group_names.end(), "mlip") > 0;
        bool file_has_ani = std::count(group_names.begin(), group_names.end(), "ani") > 0;
        if (file_has_mlip && file_has_ani) {
            throw std::runtime_error(
                "Both &mlip and &ani namelists found in " + abs_fpath
                + ", only one can be used"
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
        group_name = file_has_mlip ? "mlip" : "ani";
        if (source == "qmmm-and-mlip") {
            printWarning(
                "Reading the ML/MM options from the &" + group_name + " and &qmmm"
                " namelists is deprecated, use the &mlmm namelist instead"
            );
        }
    }

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
        {"write_xyz", "write_coords"},
        {"write_mlmm_coords", "write_coords"},
        {"write_mlmm_forces", "write_forces"},
        {"write_charges", "write_atomic_charges"},
        {"write_mlmm_atomic_charges", "write_atomic_charges"},
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
        // With mechanical embedding mm_cutoff is set internally
        std::string mlmm_coupling = mlmm_config->getString("mlmm_coupling");
        if (isMechanicalEmbedding(mlmm_coupling)) {
            if (source != "mlmm") {
                throw config_validation_error(
                    "mlmm_coupling = '" + mlmm_coupling + "' is only supported with the"
                    " &mlmm namelist"
                );
            }
            if (group.values.count("mm_cutoff") > 0) {
                throw config_validation_error(
                    "mm_cutoff can't be set with mlmm_coupling = '" + mlmm_coupling
                    + "', it is set internally"
                );
            }
            mlmm_config = mlmm_config->withValue("mm_cutoff", 0.1);
        }
        if (isGeepCoupling(mlmm_coupling) && source != "mlmm") {
            throw config_validation_error(
                "mlmm_coupling = '" + mlmm_coupling + "' is only supported with the"
                " &mlmm namelist"
            );
        }
        std::string mm_cutoff_kind = mlmm_config->getString("mm_cutoff_kind");
        std::transform(
            mm_cutoff_kind.begin(), mm_cutoff_kind.end(), mm_cutoff_kind.begin(), ::tolower
        );
        std::string supported_mm_cutoff_kind =
            isMechanicalEmbedding(mlmm_coupling) ? "none" : "hard";
        if (!mm_cutoff_kind.empty() && mm_cutoff_kind != supported_mm_cutoff_kind) {
            throw config_validation_error(
                "mm_cutoff_kind = '" + mm_cutoff_kind + "' is not supported with"
                " mlmm_coupling = '" + mlmm_coupling + "', the only supported value is '"
                + supported_mm_cutoff_kind + "'"
            );
        }
        mlmm_config = mlmm_config->withValue("mm_cutoff_kind", supported_mm_cutoff_kind);
    }
    validateConfig(mlip_config, mlmm_config);

    std::optional<ConfigDict> amber_qmmm_config;
    // With the deprecated qmmm-and-mlip, sander also reads the ML/MM config from its
    // &qmmm namelist
    if (source == "qmmm-and-mlip" && program == "sander") {
        if (!file_has_qmmm) {
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
    if (mlip_config.getInt("write_interval") < 1) {
        throw config_validation_error("write_interval must be at least 1");
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

    bool coupling_is_coulombic = mlmm_coupling == "coulombic"
        || mlmm_coupling == "coulombic-geep";
    bool coupling_is_polarizable = isPolarizableCoupling(mlmm_coupling);
    bool coupling_is_geep = isGeepCoupling(mlmm_coupling);

    if (!isOneOf(mlmm_coupling, {"custom", "none"}) && !coupling_is_coulombic
        && !coupling_is_polarizable && !isMechanicalEmbedding(mlmm_coupling)) {
        throw config_validation_error(
            "Valid mlmm_coupling values are 'coulombic', 'simple-polarizable',"
            " 'coulombic-geep', 'simple-polarizable-geep', 'custom', 'none',"
            " 'mechanical-embedding' (or 'me') and"
            " 'mechanical-embedding-no-electrostatics' (or 'me-no-electrostatics')"
        );
    }
    if (isMechanicalEmbedding(mlmm_coupling)) {
        if (use_mlip_atomic_charges) {
            throw config_validation_error(
                "mlmm_coupling = '" + mlmm_coupling + "' requires"
                " use_mlip_atomic_charges = F, the MD engine uses the topology charges"
            );
        }
        if (mlmm.getBool("use_mlip_atomic_volumes")) {
            throw config_validation_error(
                "use_mlip_atomic_volumes is not supported with mlmm_coupling = '"
                + mlmm_coupling + "'"
            );
        }
        if (use_qm_coupling) {
            throw config_validation_error(
                "use_qm_coupling is not supported with mlmm_coupling = '" + mlmm_coupling
                + "'"
            );
        }
        if (use_hybrid_mlqm) {
            throw config_validation_error(
                "use_hybrid_mlqm is not supported with mlmm_coupling = '" + mlmm_coupling
                + "'"
            );
        }
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
        if (coupling_is_coulombic && use_mlip_atomic_charges) {
            throw config_validation_error(
                "mlmm_coupling = '" + mlmm_coupling + "' is untested with"
                " use_mlip_atomic_charges = T, set allow_untested_protocols = T if you"
                " really mean to run this simulation"
            );
        }
        if (coupling_is_polarizable && !use_mlip_atomic_charges) {
            throw config_validation_error(
                "mlmm_coupling = '" + mlmm_coupling + "' is untested with fixed topology"
                " charges, set allow_untested_protocols = T if you really mean to run this"
                " simulation"
            );
        }
    }
    bool use_mlip_atomic_volumes = mlmm.getBool("use_mlip_atomic_volumes");
    bool volume_kzs_is_set = !mlmm.values("volume_kzs").empty();
    resolvedVolumeKzs(mlmm);
    if (use_mlip_atomic_volumes && !coupling_is_polarizable) {
        throw config_validation_error(
            "use_mlip_atomic_volumes = T requires mlmm_coupling = 'simple-polarizable' or"
            " 'simple-polarizable-geep'"
        );
    }
    if (!use_mlip_atomic_volumes && volume_kzs_is_set) {
        throw config_validation_error(
            "volume_kzs only has an effect with use_mlip_atomic_volumes = T"
        );
    }
    if (!inv_pol_dielectric_is_default && !coupling_is_polarizable) {
        throw config_validation_error(
            "inv_pol_dielectric only has an effect with mlmm_coupling = 'simple-polarizable'"
            " or 'simple-polarizable-geep'"
        );
    }
    resolvedGeepSigmas(mlmm);
    if (mlmm.getDouble("geep_factor") <= 0.0) {
        throw config_validation_error("geep_factor must be > 0");
    }
    if (!coupling_is_geep && !mlmm.values("geep_sigmas").empty()) {
        throw config_validation_error(
            "geep_sigmas only has an effect with mlmm_coupling = 'coulombic-geep' or"
            " 'simple-polarizable-geep'"
        );
    }
    if (!coupling_is_geep && !isDefault(mlmm, defaults, "geep_factor")) {
        throw config_validation_error(
            "geep_factor only has an effect with mlmm_coupling = 'coulombic-geep' or"
            " 'simple-polarizable-geep'"
        );
    }
    if (mlmm_coupling == "custom") {
        if (use_mlip_atomic_charges) {
            throw config_validation_error(
                "mlmm_coupling = 'custom' handles charges internally,"
                " set use_mlip_atomic_charges = F"
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
        if (coupling_is_coulombic && use_mlip_atomic_charges) {
            printWarning(
                "You are using an untested protocol: mlmm_coupling = '" + mlmm_coupling
                + "' is only tested with fixed topology charges"
            );
        }
        if (coupling_is_polarizable && !use_mlip_atomic_charges) {
            printWarning(
                "You are using an untested protocol: mlmm_coupling = '" + mlmm_coupling
                + "' is only tested with use_mlip_atomic_charges = T"
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

auto GlobalConfig::getVolumeKz(long atomic_num) const -> double {
    if (atomic_num < 1 || atomic_num > static_cast<long>(constants::NUM_ELEMENTS)) {
        throw std::out_of_range("Invalid atomic number " + std::to_string(atomic_num));
    }
    std::vector<double> kzs = resolvedVolumeKzs(mlmm());
    auto idx = static_cast<std::size_t>(atomic_num - 1);
    if (std::isnan(kzs[idx])) {
        throw std::runtime_error(
            "volume_kzs has no value for element "
            + std::string(constants::ELEMENT_SYMBOLS[idx])
            + " (atomic number " + std::to_string(atomic_num) + ")"
        );
    }
    return kzs[idx];
}

auto GlobalConfig::getGeepSigma(long atomic_num) const -> double {
    if (atomic_num < 1 || atomic_num > static_cast<long>(constants::NUM_ELEMENTS)) {
        throw std::out_of_range("Invalid atomic number " + std::to_string(atomic_num));
    }
    std::vector<double> sigmas = resolvedGeepSigmas(mlmm());
    auto idx = static_cast<std::size_t>(atomic_num - 1);
    if (std::isnan(sigmas[idx])) {
        throw std::runtime_error(
            "geep_sigmas has no value for element "
            + std::string(constants::ELEMENT_SYMBOLS[idx])
            + " (atomic number " + std::to_string(atomic_num) + ")"
        );
    }
    return sigmas[idx] * mlmm().getDouble("geep_factor");
}

auto GlobalConfig::getPolarizability(long atomic_num) const -> double {
    if (atomic_num < 1 || atomic_num > static_cast<long>(constants::NUM_ELEMENTS)) {
        throw std::out_of_range("Invalid atomic number " + std::to_string(atomic_num));
    }
    return getPolarizability(std::string(constants::ELEMENT_SYMBOLS[atomic_num - 1]));
}

auto GlobalConfig::print(const std::string& text) const -> void {
    if (!m_output_unit) {
        std::cout << text << std::flush;
        return;
    }
    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line)) {
        m_write_line(*m_output_unit, line.c_str(), static_cast<int>(line.size()));
    }
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
             "write_coords",
             "write_forces",
             "write_atomic_charges",
             "write_atomic_volumes",
             "write_atomic_alphas",
             "write_interval",
         }) {
        out << configLine(mlip(), key);
    }
    if (!getString("output_dir").empty()) {
        out << configLine(mlip(), "output_dir");
    }
    if (!getString("extra_features_file_path").empty()) {
        out << configLine(mlip(), "extra_features_file_path");
    }
    if (!m_mlmm_config) {
        out << "ML/MM:--------------end config-------------------\n";
        print(out.str());
        return;
    }
    const std::string& mlmm_coupling = getString("mlmm_coupling");
    bool use_qm_coupling = getBool("use_qm_coupling");
    const std::string& qm_coupling_method = getString("qm_coupling_method");
    out << configLine(mlmm(), "use_mlip_atomic_charges");
    out << configLine(mlmm(), "use_mlip_atomic_volumes");
    if (getBool("use_mlip_atomic_volumes") && mlmm().values("volume_kzs").empty()) {
        out << configLine(
            mlmm().withValue("volume_kzs", defaultVolumeKzsName(mlmm())), "volume_kzs"
        );
    } else if (getBool("use_mlip_atomic_volumes")) {
        out << configLine(mlmm(), "volume_kzs");
    }
    if (m_amber_qmmm_int == 1) {
        out << configLine(mlmm(), "mlmm_coupling");
    }
    if (isGeepCoupling(mlmm_coupling) && mlmm().values("geep_sigmas").empty()) {
        out << configLine(
            mlmm().withValue("geep_sigmas", std::string{"builtin"}), "geep_sigmas"
        );
    } else if (isGeepCoupling(mlmm_coupling)) {
        out << configLine(mlmm(), "geep_sigmas");
    }
    if (isGeepCoupling(mlmm_coupling)) {
        out << configLine(mlmm(), "geep_factor");
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
        print(out.str());
        return;
    }
    if (mlmm_coupling == "mechanical-embedding") {
        out << "ML/MM: Will use mechanical embedding, the QM/MM interaction energy is"
            << " calculated by the MD engine\n"
            << "     with fixed charges read from the topology file for the QM-region\n";
        print(out.str());
        return;
    }
    if (mlmm_coupling == "mechanical-embedding-no-electrostatics") {
        out << "ML/MM: Will use mechanical embedding without QM/MM electrostatic"
            << " interaction energy\n"
            << "     only the van der Waals and bonded QM/MM interactions are calculated"
            << " by the MD engine\n";
        print(out.str());
        return;
    }
    if (mlmm_coupling == "none" && !use_qm_coupling) {
        out << "ML/MM: Will use no QM/MM interaction energy, only the in vacuo ML energy\n";
        print(out.str());
        return;
    }
    out << "ML/MM: QM/MM electrostatic coupling will be calculated by MLIP Interface\n";
    if (use_qm_coupling) {
        out << "ML/MM: This is an EXPERIMENTAL feature, don't use in production code!\n"
            << "     Will delegate QM/MM electrostatic coupling to " << qm_coupling_method << "\n"
            << "     The electrostatic coupling approximation will be used\n"
            << "     The QM energies and forces will be also corrected\n";
        print(out.str());
        return;
    }
    if (mlmm_coupling == "coulombic" || mlmm_coupling == "coulombic-geep") {
        out << "ML/MM: Will use a coulombic-only QM/MM interaction energy,"
            << " with no polarization corrections\n";
    } else if (isPolarizableCoupling(mlmm_coupling)) {
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
    if (isGeepCoupling(mlmm_coupling)) {
        out << "ML/MM: The MM charges are Gaussians (GEEP), with widths (bohr) that depend"
            << " on the element of the ML atom, geep_sigmas times geep_factor\n";
    }
    if (mlmm_coupling != "custom" && getBool("use_mlip_atomic_charges")) {
        out << "ML/MM: Will use position-dependent NN-predicted charges for the QM-region,"
            << " taking into account dq/dr\n";
    } else if (mlmm_coupling != "custom") {
        out << "ML/MM: Will use fixed charges read from the topology file for the QM-region\n";
    }
    if (getBool("use_mlip_atomic_volumes")) {
        out << "ML/MM: Will use polarizabilities proportional to the NN-predicted atomic"
            << " volumes for the QM-region, alpha = kz * volume, taking into account"
            << " dvolume/dr\n";
    }
    print(out.str());
}

}  // namespace config
