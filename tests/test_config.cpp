#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "config.h"
#include "constants.h"
#include "fortran_utils.h"

// c10 defines a CHECK macro, which must not replace the one from Catch2
#undef CHECK
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

using Catch::Matchers::WithinRel;
using config::GlobalConfig;

static std::filesystem::path writeTmpFile(
    const std::string& name, const std::string& text
) {
    std::filesystem::path path = std::filesystem::temp_directory_path() / name;
    std::ofstream file(path);
    file << text;
    return path;
}

TEST_CASE("GlobalConfig getters throw before the config is read", "[config]") {
    GlobalConfig config{};
    CHECK_THROWS_AS(config.mlip(), std::logic_error);
    CHECK_THROWS_AS(config.mlmm(), std::logic_error);
}

TEST_CASE("GlobalConfig from namelist file defaults", "[config]") {
    std::filesystem::path path = writeTmpFile(
        "torchani_test_config_defaults.in", "title\n&cntrl imin=0 /\n&mlip /\n"
    );
    GlobalConfig config = GlobalConfig::fromNamelistFile(path.string());
    bool use_cuda_device = config::defaultMlipConfig().getBool("use_cuda_device");
    CHECK(
        config.device()
        == torch::Device(use_cuda_device ? torch::kCUDA : torch::kCPU, 0)
    );
    CHECK(config.dtype() == torch::kFloat64);
    CHECK_FALSE(config.useNoEvalAtoms());
    CHECK_FALSE(config.mlipInterfaceIsInit());
    CHECK(config.getBool("use_cuaev"));
    CHECK(config.getString("model_type") == "ani2x");
    CHECK_FALSE(config.getBool("use_external_nonbond"));
    CHECK(config.getInt("model_index") == -1);
    CHECK_FALSE(config.getBool("use_external_neighborlist"));
    CHECK(config.getInt("ml_charge") == 0);
    CHECK(config.getInt("ml_spin_mult") == 1);
    CHECK(config.getInt("torch_num_threads") == 1);
    CHECK(config.getString("no_eval_mask").empty());
    CHECK(config.getString("model_format") == "auto");
    CHECK_FALSE(config.getBool("skip_py_preamble"));
    CHECK_FALSE(config.getBool("do_torch_compile"));
    CHECK(config.getString("torch_compile_mode") == "default");
    std::filesystem::remove(path);
}

TEST_CASE("GlobalConfig CUDA device", "[config]") {
    std::filesystem::path path = writeTmpFile(
        "torchani_test_config_cuda.in",
        "&mlip use_cuda_device = T, cuda_device_index = 1 /\n"
    );
#ifdef TORCHANI_BUILD_CUDA
    CHECK(
        GlobalConfig::fromNamelistFile(path.string()).device()
        == torch::Device(torch::kCUDA, 1)
    );
#else
    CHECK_THROWS_AS(
        GlobalConfig::fromNamelistFile(path.string()), config::unsupported_device_error
    );
#endif
    std::filesystem::remove(path);
}

TEST_CASE("GlobalConfig from namelist file values", "[config]") {
    std::filesystem::path path = writeTmpFile(
        "torchani_test_config_values.in",
        "&mlip\n"
        "  model_type = 'animbis', use_cuda_device = F, use_double_precision = F,\n"
        "  net_charge = -1, no_eval_mask = ':WAT', use_amber_neighborlist = T,\n"
        "  use_all_amber_nonbond = T, use_cuaev = F, model_index = 3, cuda_device_index = 1,\n"
        "  ml_spin_mult = 2, extra_features_file_path = 'features.txt', torch_num_threads = 4,\n"
        "/\n"
        "&mlip model_type = 'ignored' /\n"
    );
    GlobalConfig config = GlobalConfig::fromNamelistFile(path.string());
    CHECK(config.getInt("cuda_device_index") == 1);
    CHECK(config.getInt("torch_num_threads") == 4);
    CHECK(config.dtype() == torch::kFloat32);
    // Only mlip_set_no_eval_atoms_mask enables the no-eval atoms
    CHECK_FALSE(config.useNoEvalAtoms());
    CHECK_FALSE(config.getBool("use_cuaev"));
    CHECK(config.getString("model_type") == "animbis");
    CHECK(config.getBool("use_external_nonbond"));
    CHECK(config.getInt("model_index") == 3);
    CHECK(config.getBool("use_external_neighborlist"));
    CHECK(config.getInt("ml_charge") == -1);
    CHECK(config.getInt("ml_spin_mult") == 2);
    CHECK(config.getString("no_eval_mask") == ":WAT");
    CHECK(config.getString("extra_features_file_path") == "features.txt");
    std::filesystem::remove(path);

    path =
        writeTmpFile("torchani_test_config_values.in", "&mlip use_cuda_device = F /\n");
    CHECK(
        GlobalConfig::fromNamelistFile(path.string()).device()
        == torch::Device(torch::kCPU, 0)
    );

    writeTmpFile(
        "torchani_test_config_values.in",
        "&mlip ml_charge = 2, use_external_neighborlist = T /\n"
    );
    config = GlobalConfig::fromNamelistFile(path.string());
    CHECK(config.getInt("ml_charge") == 2);
    CHECK(config.getBool("use_external_neighborlist"));
    std::filesystem::remove(path);
}

TEST_CASE("GlobalConfig from namelist file errors", "[config]") {
    CHECK_THROWS_AS(
        GlobalConfig::fromNamelistFile("/nonexistent/torchani_test.in"),
        std::runtime_error
    );
    std::filesystem::path path =
        writeTmpFile("torchani_test_config_errors.in", "&cntrl /\n");
    CHECK_THROWS_AS(GlobalConfig::fromNamelistFile(path.string()), std::runtime_error);

    writeTmpFile("torchani_test_config_errors.in", "&mlip net_charge = 1.0 /\n");
    CHECK_THROWS_AS(
        GlobalConfig::fromNamelistFile(path.string()),
        fortran_utils::namelist_type_error
    );
    writeTmpFile("torchani_test_config_errors.in", "&mlip spin_mult = 2 /\n");
    CHECK_THROWS_AS(
        GlobalConfig::fromNamelistFile(path.string()),
        fortran_utils::namelist_unknown_key_error
    );
    writeTmpFile(
        "torchani_test_config_errors.in", "&mlip net_charge = 1, ml_charge = 1 /\n"
    );
    CHECK_THROWS_AS(
        GlobalConfig::fromNamelistFile(path.string()),
        fortran_utils::namelist_alias_conflict_error
    );
    writeTmpFile(
        "torchani_test_config_errors.in",
        "&mlip model_type = 'nutmeg-small::features.txt' /\n"
    );
    CHECK_THROWS_AS(
        GlobalConfig::fromNamelistFile(path.string()), config::config_validation_error
    );
    writeTmpFile("torchani_test_config_errors.in", "&mlip torch_num_threads = 0 /\n");
    CHECK_THROWS_AS(
        GlobalConfig::fromNamelistFile(path.string()), config::config_validation_error
    );
    writeTmpFile("torchani_test_config_errors.in", "&mlip /\n&ani /\n");
    CHECK_THROWS_AS(GlobalConfig::fromNamelistFile(path.string()), std::runtime_error);
    std::filesystem::remove(path);
}

TEST_CASE("GlobalConfig model format and Python model options", "[config]") {
    const std::string fname = "torchani_test_config_model_format.in";
    std::filesystem::path path = writeTmpFile(
        fname,
        "&mlip model_type = '/path/to/model', model_format = 'python',\n"
        "  skip_py_preamble = T, do_torch_compile = T,\n"
        "  torch_compile_mode = 'max-autotune-no-cudagraphs' /\n"
    );
    GlobalConfig config = GlobalConfig::fromNamelistFile(path.string());
    CHECK(config.getString("model_format") == "python");
    CHECK(config.getBool("skip_py_preamble"));
    CHECK(config.getBool("do_torch_compile"));
    CHECK(config.getString("torch_compile_mode") == "max-autotune-no-cudagraphs");

    writeTmpFile(
        fname, "&mlip model_type = '/path/to/model', model_format = 'pt2' /\n"
    );
    CHECK_THROWS_AS(
        GlobalConfig::fromNamelistFile(path.string()), config::config_validation_error
    );
    writeTmpFile(fname, "&mlip model_type = 'ani2x', model_format = 'torchscript' /\n");
    CHECK_THROWS_WITH(
        GlobalConfig::fromNamelistFile(path.string()),
        Catch::Matchers::ContainsSubstring("built-in model")
    );
    writeTmpFile(
        fname, "&mlip do_torch_compile = T, torch_compile_mode = 'fastest' /\n"
    );
    CHECK_THROWS_AS(
        GlobalConfig::fromNamelistFile(path.string()), config::config_validation_error
    );
    writeTmpFile(fname, "&mlip torch_compile_mode = 'reduce-overhead' /\n");
    CHECK_THROWS_WITH(
        GlobalConfig::fromNamelistFile(path.string()),
        Catch::Matchers::ContainsSubstring("requires do_torch_compile")
    );
    std::filesystem::remove(path);
}

TEST_CASE("GlobalConfig from deprecated ani namelist", "[config]") {
    std::filesystem::path path = writeTmpFile(
        "torchani_test_config_ani.in", "&cntrl /\n&ani model_type = 'animbis' /\n"
    );
    std::stringstream captured;
    std::streambuf* cerr_buf = std::cerr.rdbuf(captured.rdbuf());
    GlobalConfig config = GlobalConfig::fromNamelistFile(path.string());
    std::cerr.rdbuf(cerr_buf);
    CHECK(config.getString("model_type") == "animbis");
    CHECK(captured.str().find("&ani namelist") != std::string::npos);
    CHECK(captured.str().find("use &mlip instead") != std::string::npos);
    std::filesystem::remove(path);
}

TEST_CASE("GlobalConfig ML/MM mm_cutoff, ml_mask and ml_shake", "[config]") {
    std::filesystem::path path = writeTmpFile(
        "torchani_test_config_mlmm_cutoff_mask.in",
        "&mlmm mm_cutoff = 10.0, ml_mask = ':1-2', ml_shake = F /\n"
    );
    GlobalConfig config =
        GlobalConfig::fromNamelistFile(path.string(), "mlmm", "sander");
    CHECK(config.getDouble("mm_cutoff") == 10.0);
    CHECK(config.getString("ml_mask") == ":1-2");
    CHECK_FALSE(config.getBool("ml_shake"));
    std::filesystem::remove(path);
}

TEST_CASE("GlobalConfig ML/MM electrostatics_kind", "[config]") {
    // The default is the first supported value
    auto [mlmm_coupling, supported_kinds, unsupported_kind] = GENERATE(
        std::make_tuple(
            std::string{"coulombic"},
            std::vector<std::string>{"hard-cutoff", "ewald"},
            std::string{"none"}
        ),
        std::make_tuple(
            std::string{"coulombic-geep"},
            std::vector<std::string>{"hard-cutoff", "ewald"},
            std::string{"none"}
        ),
        std::make_tuple(
            std::string{"none"},
            std::vector<std::string>{"hard-cutoff", "ewald"},
            std::string{"none"}
        ),
        std::make_tuple(
            std::string{"me"},
            std::vector<std::string>{"none"},
            std::string{"hard-cutoff"}
        ),
        std::make_tuple(
            std::string{"me-no-electrostatics"},
            std::vector<std::string>{"none"},
            std::string{"ewald"}
        )
    );
    const std::string fname = "torchani_test_config_electrostatics_kind.in";
    const std::string coupling_line = "&mlmm mlmm_coupling = '" + mlmm_coupling + "'";
    std::filesystem::path path = writeTmpFile(fname, coupling_line + " /\n");
    GlobalConfig config =
        GlobalConfig::fromNamelistFile(path.string(), "mlmm", "sander");
    CHECK(config.getString("electrostatics_kind") == supported_kinds.front());
    for (const std::string& kind : supported_kinds) {
        // Case-insensitive
        std::string upper_kind = kind;
        std::transform(
            upper_kind.begin(), upper_kind.end(), upper_kind.begin(), ::toupper
        );
        writeTmpFile(
            fname, coupling_line + ", electrostatics_kind = '" + upper_kind + "' /\n"
        );
        config = GlobalConfig::fromNamelistFile(path.string(), "mlmm", "sander");
        INFO(kind);
        CHECK(config.getString("electrostatics_kind") == kind);
    }
    for (
        const std::string& kind :
        {unsupported_kind, std::string{"hard"}, std::string{"switched"}}
    ) {
        writeTmpFile(
            fname, coupling_line + ", electrostatics_kind = '" + kind + "' /\n"
        );
        INFO(kind);
        CHECK_THROWS_AS(
            GlobalConfig::fromNamelistFile(path.string(), "mlmm", "sander"),
            config::config_validation_error
        );
    }
    std::filesystem::remove(path);
}

TEST_CASE("GlobalConfig ML/MM electrostatics_kind ewald requires mlmm", "[config]") {
    const std::string fname = "torchani_test_config_electrostatics_kind_qmmm.in";
    const std::string qmmm_line =
        "&qmmm qmmask = ':1', qm_theory = 'EXTERN', qmcut = 8.0 /\n";
    std::filesystem::path path = writeTmpFile(
        fname, "&mlip electrostatics_kind = 'hard-cutoff' /\n" + qmmm_line
    );
    GlobalConfig config =
        GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip", "sander");
    CHECK(config.getString("electrostatics_kind") == "hard-cutoff");
    writeTmpFile(fname, "&mlip electrostatics_kind = 'ewald' /\n" + qmmm_line);
    CHECK_THROWS_AS(
        GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip", "sander"),
        config::config_validation_error
    );
    std::filesystem::remove(path);
}

TEST_CASE("GlobalConfig ML/MM mechanical embedding", "[config]") {
    auto [alias, mlmm_coupling] = GENERATE(
        std::make_pair(std::string{"ME"}, std::string{"mechanical-embedding"}),
        std::make_pair(
            std::string{"ME-no-electrostatics"},
            std::string{"mechanical-embedding-no-electrostatics"}
        )
    );
    const std::string fname = "torchani_test_config_mechanical_embedding.in";
    std::filesystem::path path =
        writeTmpFile(fname, "&mlmm mlmm_coupling = '" + alias + "' /\n");
    GlobalConfig config =
        GlobalConfig::fromNamelistFile(path.string(), "mlmm", "sander");
    CHECK(config.getString("mlmm_coupling") == mlmm_coupling);
    CHECK(config.getDouble("mm_cutoff") == 0.1);
    writeTmpFile(fname, "&mlmm mlmm_coupling = '" + mlmm_coupling + "' /\n");
    config = GlobalConfig::fromNamelistFile(path.string(), "mlmm", "sander");
    CHECK(config.getString("mlmm_coupling") == mlmm_coupling);

    const std::string coupling_line = "&mlmm mlmm_coupling = '" + mlmm_coupling + "', ";
    const std::vector<std::string> invalid_texts = {
        coupling_line + "mm_cutoff = 0.1 /\n",
        coupling_line + "use_mlip_atomic_charges = T, allow_untested_protocols = T /\n",
        coupling_line + "use_mlip_atomic_volumes = T /\n",
        coupling_line + "use_qm_coupling = T, qm_coupling_method = 'orca' /\n",
        coupling_line + "use_hybrid_mlqm = T, hybrid_mlqm_method = 'orca' /\n",
    };
    for (const std::string& text : invalid_texts) {
        writeTmpFile(fname, text);
        INFO(text);
        CHECK_THROWS_AS(
            GlobalConfig::fromNamelistFile(path.string(), "mlmm", "sander"),
            config::config_validation_error
        );
    }
    // Only supported with the &mlmm namelist
    writeTmpFile(fname, "&mlip mlmm_coupling = '" + alias + "' /\n");
    CHECK_THROWS_AS(
        GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip", "none"),
        config::config_validation_error
    );
    std::filesystem::remove(path);
}

TEST_CASE("GlobalConfig from_namelist", "[config]") {
    const std::string fname = "torchani_test_config_from_namelist.in";
    std::filesystem::path path = writeTmpFile(
        fname, "&mlip model_type = 'ani1x' /\n&mlmm model_type = 'animbis' /\n"
    );
    // "mlmm" reads &mlmm, case-insensitive
    GlobalConfig config =
        GlobalConfig::fromNamelistFile(path.string(), "MLMM", "sander");
    CHECK(config.getString("model_type") == "animbis");
    CHECK(config.getInt("ml_charge") == 0);
    // "mlip" and "qmmm-and-mlip" read &mlip, only "qmmm-and-mlip" warns
    std::stringstream captured;
    std::streambuf* cerr_buf = std::cerr.rdbuf(captured.rdbuf());
    config = GlobalConfig::fromNamelistFile(path.string(), "mlip", "none");
    CHECK(captured.str().empty());
    config = GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip", "none");
    std::cerr.rdbuf(cerr_buf);
    CHECK(config.getString("model_type") == "ani1x");
    CHECK(captured.str().find("use the &mlmm namelist instead") != std::string::npos);

    CHECK(config.hasMlmmConfig());
    config = GlobalConfig::fromNamelistFile(path.string(), "mlip", "sander");
    CHECK_FALSE(config.hasMlmmConfig());
    CHECK_THROWS_AS(
        GlobalConfig::fromNamelistFile(path.string(), "qmmm", "sander"),
        std::runtime_error
    );
    writeTmpFile(fname, "&mlip /\n");
    CHECK_THROWS_AS(
        GlobalConfig::fromNamelistFile(path.string(), "mlmm", "sander"),
        std::runtime_error
    );
    // &qmmm can't be used with &mlmm, for any calling program
    writeTmpFile(fname, "&mlmm /\n&qmmm qmcharge = 1 /\n");
    CHECK_THROWS_AS(
        GlobalConfig::fromNamelistFile(path.string(), "mlmm", "sander"),
        config::config_validation_error
    );
    CHECK_THROWS_AS(
        GlobalConfig::fromNamelistFile(path.string(), "mlmm", "none"),
        config::config_validation_error
    );
    std::filesystem::remove(path);
}

TEST_CASE("GlobalConfig ML/MM config", "[config]") {
    std::filesystem::path path = writeTmpFile(
        "torchani_test_config_mlmm.in",
        "&mlip mlmm_coupling = 1, use_torchani_charges = T, inv_pol_dielectric = 1,\n"
        "  pol_H = 2.5, pol_Ar = 10.0, pol_Fe = 50.0, write_xyz = T, switching_program = 'orca',\n"
        "  use_switching_function = T,\n"
        "  qlow = 0.1, use_torch_coupling = T /\n"
        "&qmmm qmmask = ':1', qm_theory = 'EXTERN', qmcharge = 0, qm_ewald = 0, spin = 2,\n"
        "  qmcut = 8.0 /\n"
    );
    std::stringstream captured_int;
    std::streambuf* int_cerr_buf = std::cerr.rdbuf(captured_int.rdbuf());
    GlobalConfig config =
        GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip", "sander");
    std::cerr.rdbuf(int_cerr_buf);
    CHECK(
        captured_int.str().find("use mlmm_coupling = 'simple-polarizable'")
        != std::string::npos
    );
    CHECK(config.getString("mlmm_coupling") == "simple-polarizable");
    CHECK(config.getBool("use_mlip_atomic_charges"));
    CHECK_FALSE(config.getBool("use_numerical_mlmm_forces"));
    CHECK_FALSE(config.getBool("allow_untested_protocols"));
    CHECK(config.getDouble("inv_pol_dielectric") == 1.0);
    CHECK(config.getDouble("mm_cutoff") == 15.0);
    CHECK(config.getString("ml_mask").empty());
    CHECK(config.getBool("ml_shake"));
    CHECK(config.getDouble("pol_h") == 2.5);
    CHECK(config.getDouble("pol_c") == 11.3);
    CHECK(config.getDouble("pol_ar") == 10.0);
    CHECK(config.getDouble("pol_fe") == 50.0);
    CHECK(std::isnan(config.getDouble("pol_og")));
    CHECK(config.getBool("write_coords"));
    CHECK_FALSE(config.getBool("write_forces"));
    CHECK(config.getBool("use_hybrid_mlqm"));
    CHECK(config.getString("hybrid_mlqm_method") == "orca");
    CHECK(config.getDouble("hybrid_mlqm_switch_low") == 0.1);
    CHECK(config.getDouble("hybrid_mlqm_switch_high") == 0.3);
    CHECK_FALSE(config.getBool("use_qm_coupling"));
    CHECK(config.getString("qm_coupling_method") == "none");

    CHECK_THROWS_AS(
        GlobalConfig::fromNamelistFile(path.string()),
        fortran_utils::namelist_unknown_key_error
    );
    writeTmpFile(
        "torchani_test_config_mlmm.in",
        "&mlip use_extcoupling = T, extcoupling_program = 'orca', mlmm_coupling = 'none' /\n"
    );
    config = GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip");
    CHECK(config.getBool("use_qm_coupling"));
    CHECK(config.getString("qm_coupling_method") == "orca");
    writeTmpFile("torchani_test_config_mlmm.in", "&mlip /\n");
    CHECK_THROWS_AS(
        GlobalConfig::fromNamelistFile(path.string()).mlmm(), std::logic_error
    );
    config = GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip");
    CHECK(config.getString("mlmm_coupling") == "coulombic");

    writeTmpFile(
        "torchani_test_config_mlmm.in",
        "&mlip mlmm_coupling = 'Simple-Polarizable', use_mlip_atomic_charges = T,\n"
        "  write_charges = T, write_charges_grad = T, write_forces = T,\n"
        "  use_hybrid_mlqm = T, hybrid_mlqm_method = 'gau', hybrid_mlqm_switch_high = 0.5 /\n"
    );
    config = GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip");
    CHECK(config.getString("mlmm_coupling") == "simple-polarizable");
    CHECK(config.getBool("use_mlip_atomic_charges"));
    CHECK(config.getBool("write_atomic_charges"));
    CHECK_FALSE(config.getBool("write_coords"));
    CHECK(config.getBool("write_forces"));
    CHECK(config.getBool("use_hybrid_mlqm"));
    CHECK(config.getString("hybrid_mlqm_method") == "gau");
    CHECK(config.getDouble("hybrid_mlqm_switch_high") == 0.5);
    CHECK(config.getString("qm_coupling_method") == "none");
    std::filesystem::remove(path);
}

TEST_CASE("GlobalConfig ML/MM config validation", "[config]") {
    const std::string fname = "torchani_test_config_validation.in";
    const std::vector<std::string> invalid_texts{
        "&mlip use_external_neighborlist = T /\n",
        "&mlip mlmm_coupling = 4 /\n",
        "&mlip mlmm_coupling = -1 /\n",
        "&mlip mlmm_coupling = 'polarizable' /\n",
        "&mlip mlmm_coupling = 'coulombic', use_mlip_atomic_charges = T /\n",
        "&mlip mlmm_coupling = 'simple-polarizable' /\n",
        "&mlip inv_pol_dielectric = 1.0 /\n",
        "&mlip mlmm_coupling = 'none', inv_pol_dielectric = 1.0 /\n",
        "&mlip mlmm_coupling = 'custom', use_qm_coupling = T, qm_coupling_method = 'orca' /\n",
        "&mlip mlmm_coupling = 'custom', use_mlip_atomic_charges = T,\n"
        "  allow_untested_protocols = T /\n",
        "&mlip write_interval = 0 /\n",
        "&mlip mlmm_coupling = 'coulombic', use_mlip_atomic_volumes = T,\n"
        "  volume_kzs = 1.0 /\n",
        "&mlip mlmm_coupling = 'simple-polarizable', use_mlip_atomic_charges = T,\n"
        "  volume_kzs = 1.0 /\n",
        "&mlip mlmm_coupling = 'simple-polarizable', use_mlip_atomic_charges = T,\n"
        "  use_mlip_atomic_volumes = T, volume_kzs = 'builtin-nonexistent' /\n",
        "&mlip mlmm_coupling = 'simple-polarizable', use_mlip_atomic_charges = T,\n"
        "  use_mlip_atomic_volumes = T, volume_kzs = 1.0, 'builtin-mbis-charges' /\n",
        "&mlip use_qm_coupling = T /\n",
        "&mlip use_qm_coupling = T, qm_coupling_method = 'orca' /\n",
        "&mlip use_hybrid_mlqm = T /\n",
        "&mlip use_hybrid_mlqm = T, hybrid_mlqm_method = 'orca', mlmm_coupling = 'custom' /\n",
        "&mlip use_hybrid_mlqm = T, hybrid_mlqm_method = 'orca', use_qm_coupling = T,\n"
        "  qm_coupling_method = 'orca' /\n",
        "&mlip qm_coupling_method = 'orca' /\n",
        "&mlip hybrid_mlqm_method = 'orca' /\n",
        "&mlip use_numerical_mlmm_forces = T /\n",
        "&mlip use_charges_derivatives = F /\n",
        "&mlip use_numerical_qmmm_forces = T /\n",
    };
    for (const std::string& text : invalid_texts) {
        std::filesystem::path path = writeTmpFile(fname, text);
        CHECK_THROWS_AS(
            GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip"),
            config::config_validation_error
        );
        std::filesystem::remove(path);
    }

    std::filesystem::path path = writeTmpFile(
        fname,
        "&mlip mlmm_coupling = 'coulombic', use_mlip_atomic_charges = T,\n"
        "  allow_untested_protocols = T /\n"
    );
    std::stringstream captured;
    std::streambuf* cerr_buf = std::cerr.rdbuf(captured.rdbuf());
    GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip");
    std::cerr.rdbuf(cerr_buf);
    CHECK(captured.str().find("untested protocol") != std::string::npos);
    std::filesystem::remove(path);
}

TEST_CASE("GlobalConfig deprecated no-op options", "[config]") {
    std::filesystem::path path = writeTmpFile(
        "torchani_test_config_torch_coupling.in",
        "&mlip use_torch_coupling = F, write_charges_grad = T /\n"
    );
    std::stringstream captured;
    std::streambuf* cerr_buf = std::cerr.rdbuf(captured.rdbuf());
    GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip");
    std::cerr.rdbuf(cerr_buf);
    CHECK(captured.str().find("use_torch_coupling is deprecated") != std::string::npos);
    CHECK(captured.str().find("write_charges_grad is deprecated") != std::string::npos);
    CHECK_THROWS_AS(
        GlobalConfig::fromNamelistFile(path.string()),
        fortran_utils::namelist_unknown_key_error
    );
    std::filesystem::remove(path);
}

TEST_CASE("GlobalConfig Amber &qmmm namelist", "[config]") {
    const std::string fname = "torchani_test_config_qmmm.in";
    std::filesystem::path path = writeTmpFile(fname, "&mlip /\n");
    CHECK_THROWS_AS(
        GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip", "sander"),
        std::runtime_error
    );
    CHECK_NOTHROW(GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip"));
    writeTmpFile(fname, "&mlip /\n&qmmm qmmm_int = 2 /\n");
    CHECK_NOTHROW(GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip"));
    writeTmpFile(
        fname,
        "&mlip use_qm_coupling = T, qm_coupling_method = 'orca', mlmm_coupling = 'none' /\n"
        "&qmmm qmcharge = 1 /\n"
    );
    CHECK_NOTHROW(
        GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip", "sander")
    );
    writeTmpFile(
        fname,
        "&mlip use_qm_coupling = T, qm_coupling_method = 'orca', use_mlip_atomic_charges = T,\n"
        "  mlmm_coupling = 'none', allow_untested_protocols = T /\n&qmmm qmcharge = 1 /\n"
    );
    std::stringstream captured_charges;
    std::streambuf* charges_cerr_buf = std::cerr.rdbuf(captured_charges.rdbuf());
    CHECK_NOTHROW(
        GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip", "sander")
    );
    std::cerr.rdbuf(charges_cerr_buf);
    CHECK(
        captured_charges.str().find("use_mlip_atomic_charges has no effect")
        != std::string::npos
    );
    writeTmpFile(fname, "&mlip /\n&qmmm qmcharge = 'a' /\n");
    CHECK_THROWS_AS(
        GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip", "sander"),
        fortran_utils::namelist_type_error
    );

    const std::vector<std::string> invalid_texts{
        "&mlip allow_untested_protocols = T /\n&qmmm qmmm_int = 2 /\n",
        "&mlip mlmm_coupling = 'none' /\n&qmmm qmmm_int = 0 /\n",
        "&mlip allow_untested_protocols = T /\n&qmmm qmmm_int = 0 /\n",
        "&mlip allow_untested_protocols = T /\n&qmmm qmmm_int = 5 /\n",
        "&mlip mlmm_coupling = 'simple-polarizable', use_mlip_atomic_charges = T,\n"
        "  allow_untested_protocols = T /\n&qmmm qmmm_int = 0 /\n",
        "&mlip mlmm_coupling = 'simple-polarizable', use_mlip_atomic_charges = T,\n"
        "  allow_untested_protocols = T /\n&qmmm qmmm_int = 5 /\n",
        "&mlip mlmm_coupling = 'none', use_qm_coupling = T, qm_coupling_method = 'orca',\n"
        "  allow_untested_protocols = T /\n&qmmm qmmm_int = 0 /\n",
        "&mlip mlmm_coupling = 'none', use_qm_coupling = T, qm_coupling_method = 'orca',\n"
        "  allow_untested_protocols = T /\n&qmmm qmmm_int = 5 /\n",
    };
    for (const std::string& text : invalid_texts) {
        writeTmpFile(fname, text);
        CHECK_THROWS_AS(
            GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip", "sander"),
            config::config_validation_error
        );
    }

    writeTmpFile(
        fname,
        "&mlip mlmm_coupling = 'none', allow_untested_protocols = T /\n&qmmm qmmm_int = 0 /\n"
    );
    std::stringstream captured;
    std::streambuf* cerr_buf = std::cerr.rdbuf(captured.rdbuf());
    GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip", "sander");
    std::cerr.rdbuf(cerr_buf);
    CHECK(captured.str().find("qmmm_int = 0") != std::string::npos);
    std::filesystem::remove(path);
}

static std::string capturedPrintInfo(const GlobalConfig& config) {
    std::stringstream captured;
    std::streambuf* cout_buf = std::cout.rdbuf(captured.rdbuf());
    config.printInfo();
    std::cout.rdbuf(cout_buf);
    return captured.str();
}

TEST_CASE("GlobalConfig printInfo", "[config]") {
    const std::string fname = "torchani_test_config_print.in";
    std::filesystem::path path =
        writeTmpFile(fname, "&mlip model_type = 'animbis' /\n");
    std::string info = capturedPrintInfo(GlobalConfig::fromNamelistFile(path.string()));
    CHECK(info.find("model_type") != std::string::npos);
    CHECK(info.find("animbis") != std::string::npos);
    CHECK(info.find("mlmm_coupling") == std::string::npos);
    CHECK(info.find("extra_features_file_path") == std::string::npos);

    writeTmpFile(fname, "&mlip extra_features_file_path = 'features.txt' /\n");
    info = capturedPrintInfo(GlobalConfig::fromNamelistFile(path.string()));
    CHECK(info.find("extra_features_file_path") != std::string::npos);
    CHECK(info.find("features.txt") != std::string::npos);

    writeTmpFile(
        fname,
        "&mlip mlmm_coupling = 'simple-polarizable', use_mlip_atomic_charges = T,\n"
        "  write_xyz = T /\n&qmmm /\n"
    );
    info = capturedPrintInfo(
        GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip", "sander")
    );
    CHECK(info.find("write_coords") != std::string::npos);
    CHECK(info.find("write_xyz") == std::string::npos);
    CHECK(info.find("simple-polarizable") != std::string::npos);
    CHECK(info.find("calculated by MLIP Interface") != std::string::npos);
    CHECK(info.find("NN-predicted charges") != std::string::npos);
    CHECK(info.find("INTERNAL OPTIONS") == std::string::npos);

    writeTmpFile(
        fname,
        "&mlip mlmm_coupling = 'none', allow_untested_protocols = T,\n"
        "  hybrid_mlqm_switch_low = 0.1 /\n"
        "&qmmm qmmm_int = 0 /\n"
    );
    std::stringstream captured_cerr;
    std::streambuf* cerr_buf = std::cerr.rdbuf(captured_cerr.rdbuf());
    info = capturedPrintInfo(
        GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip", "sander")
    );
    std::cerr.rdbuf(cerr_buf);
    CHECK(info.find("INTERNAL OPTIONS") != std::string::npos);
    CHECK(info.find("hybrid_mlqm_switch_low") != std::string::npos);
    CHECK(info.find("mlmm_coupling") == std::string::npos);
    CHECK(info.find("calculated by MLIP Interface") == std::string::npos);

    writeTmpFile(fname, "&mlip mlmm_coupling = 'none' /\n&qmmm /\n");
    info = capturedPrintInfo(
        GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip", "sander")
    );
    CHECK(info.find("only the in vacuo ML energy") != std::string::npos);
    CHECK(info.find("calculated by MLIP Interface") == std::string::npos);
    std::filesystem::remove(path);
}

static std::vector<std::pair<int, std::string>> written_lines;

static void writeLine(int unit, const char* line, int line_len) {
    written_lines.emplace_back(unit, std::string(line, line_len));
}

TEST_CASE("GlobalConfig print to output unit", "[config]") {
    config::ConfigDict mlip_config = config::defaultMlipConfig();
    mlip_config = mlip_config.withValue("use_cuda_device", false);
    GlobalConfig config{mlip_config};
    written_lines.clear();
    config.setOutputUnit(116, writeLine);
    std::stringstream captured;
    std::streambuf* cout_buf = std::cout.rdbuf(captured.rdbuf());
    config.print("\nfirst\nsecond\n");
    std::cout.rdbuf(cout_buf);
    CHECK(captured.str().empty());
    std::vector<std::pair<int, std::string>> expect_lines = {
        {116, ""}, {116, "first"}, {116, "second"}
    };
    CHECK(written_lines == expect_lines);
}

TEST_CASE("GlobalConfig from config dicts", "[config]") {
    config::ConfigDict mlip_config = config::defaultMlipConfig();
    mlip_config = mlip_config.withValue("use_cuda_device", false);
    mlip_config = mlip_config.withValue("use_double_precision", false);
    mlip_config = mlip_config.withValue("no_eval_mask", std::string{"  "});
    GlobalConfig config{mlip_config};
    CHECK(config.device() == torch::Device(torch::kCPU, 0));
    CHECK(config.dtype() == torch::kFloat32);
    CHECK_FALSE(config.useNoEvalAtoms());
    CHECK_THROWS_AS(config.mlmm(), std::logic_error);

    mlip_config = mlip_config.withValue("no_eval_mask", std::string{":WAT"});
    CHECK_FALSE(GlobalConfig{mlip_config}.useNoEvalAtoms());
}

TEST_CASE("Config dict getters", "[config]") {
    config::ConfigDict mlip_config = config::defaultMlipConfig();
    CHECK(mlip_config.getInt("ml_spin_mult") == 1);
    CHECK_THROWS_AS(mlip_config.getBool("ml_spin_mult"), std::bad_variant_access);
    CHECK_THROWS_AS(mlip_config.getInt("no_such_key"), std::out_of_range);
    mlip_config = mlip_config.withValue("ml_charge", 1L << 40);
    CHECK_THROWS_AS(mlip_config.getInt("ml_charge"), std::out_of_range);
}

TEST_CASE("GlobalConfig polarizabilities", "[config]") {
    config::ConfigDict mlip_config = config::defaultMlipConfig();
    mlip_config = mlip_config.withValue("use_cuda_device", false);
    CHECK_THROWS_AS(GlobalConfig{mlip_config}.getPolarizability(6L), std::logic_error);

    config::ConfigDict mlmm_config = config::defaultMlmmConfig();
    mlmm_config = mlmm_config.withValue("pol_c", 12.5);
    GlobalConfig config{mlip_config, mlmm_config};
    CHECK(config.getPolarizability(6L) == 12.5);
    CHECK(config.getPolarizability("C") == 12.5);
    CHECK(config.getPolarizability("c") == 12.5);
    CHECK(config.getPolarizability(1L) == 3.08);
    // Elements past Ar have no default
    CHECK_THROWS_AS(config.getPolarizability(19L), std::runtime_error);
    CHECK_THROWS_AS(config.getPolarizability(0L), std::out_of_range);
    CHECK_THROWS_AS(config.getPolarizability(119L), std::out_of_range);
    CHECK_THROWS_AS(config.getPolarizability("Xx"), std::out_of_range);
}

TEST_CASE("GlobalConfig volume kzs", "[config]") {
    config::ConfigDict mlip_config = config::defaultMlipConfig();
    mlip_config = mlip_config.withValue("use_cuda_device", false);
    CHECK_THROWS_AS(GlobalConfig{mlip_config}.getVolumeKz(6L), std::logic_error);

    std::filesystem::path path = writeTmpFile(
        "torchani_test_config_volume_kzs.in",
        "&mlip mlmm_coupling = 'simple-polarizable', use_mlip_atomic_charges = T,\n"
        "  use_mlip_atomic_volumes = T, volume_kzs = 0.5, , 1 /\n"
    );
    GlobalConfig config =
        GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip");
    CHECK(config.getVolumeKz(1L) == 0.5);
    CHECK(config.getVolumeKz(3L) == 1.0);
    CHECK_THROWS_AS(config.getVolumeKz(2L), std::runtime_error);
    CHECK_THROWS_AS(config.getVolumeKz(4L), std::runtime_error);
    CHECK_THROWS_AS(config.getVolumeKz(0L), std::out_of_range);

    writeTmpFile(
        "torchani_test_config_volume_kzs.in",
        "&mlip mlmm_coupling = 'simple-polarizable', use_mlip_atomic_charges = T,\n"
        "  use_mlip_atomic_volumes = T, volume_kzs = 'builtin-mbis-charges' /\n"
    );
    config = GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip");
    CHECK(config.getVolumeKz(1L) == 0.0);
    CHECK(config.getVolumeKz(6L) == 0.7190062581351027);
    CHECK(config.getVolumeKz(16L) == 0.0);
    CHECK(config.getVolumeKz(17L) == 0.47491014980647467);
    CHECK_THROWS_AS(config.getVolumeKz(2L), std::runtime_error);
    CHECK_THROWS_AS(config.getVolumeKz(9L), std::runtime_error);
    CHECK_THROWS_AS(config.getVolumeKz(18L), std::runtime_error);

    writeTmpFile(
        "torchani_test_config_volume_kzs.in",
        "&mlip mlmm_coupling = 'simple-polarizable', use_mlip_atomic_charges = T,\n"
        "  use_mlip_atomic_volumes = T /\n"
    );
    config = GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip");
    CHECK(config.getVolumeKz(1L) == 0.0);
    CHECK(config.getVolumeKz(6L) == 0.7190062581351027);

    writeTmpFile(
        "torchani_test_config_volume_kzs.in",
        "&mlip mlmm_coupling = 'simple-polarizable', allow_untested_protocols = T,\n"
        "  use_mlip_atomic_volumes = T /\n"
    );
    config = GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip");
    CHECK(config.getVolumeKz(1L) == 0.7659661617582175);
    CHECK(config.getVolumeKz(6L) == 0.36055843774225754);

    writeTmpFile(
        "torchani_test_config_volume_kzs.in",
        "&mlip mlmm_coupling = 'simple-polarizable', use_mlip_atomic_charges = T,\n"
        "  use_mlip_atomic_volumes = T, volume_kzs = 'builtin-fixed-resp-charges' /\n"
    );
    config = GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip");
    CHECK(config.getVolumeKz(1L) == 0.7659661617582175);
    CHECK(config.getVolumeKz(16L) == 0.030818228617190933);
    CHECK_THROWS_AS(config.getVolumeKz(3L), std::runtime_error);
    std::filesystem::remove(path);
}

TEST_CASE("GlobalConfig GEEP sigmas", "[config]") {
    config::ConfigDict mlip_config = config::defaultMlipConfig();
    mlip_config = mlip_config.withValue("use_cuda_device", false);
    CHECK_THROWS_AS(GlobalConfig{mlip_config}.getGeepSigma(6L), std::logic_error);

    const std::string fname = "torchani_test_config_geep_sigmas.in";
    std::filesystem::path path =
        writeTmpFile(fname, "&mlmm mlmm_coupling = 'coulombic-geep' /\n");
    GlobalConfig config =
        GlobalConfig::fromNamelistFile(path.string(), "mlmm", "sander");
    // The builtin widths are in Angstrom, converted to bohr
    CHECK_THAT(
        config.getGeepSigma(1L),
        WithinRel(1.00 * constants::CODATA08_ANGSTROM_TO_BOHR, 1e-12)
    );
    CHECK_THAT(
        config.getGeepSigma(6L),
        WithinRel(1.50 * constants::CODATA08_ANGSTROM_TO_BOHR, 1e-12)
    );
    CHECK_THAT(
        config.getGeepSigma(54L),
        WithinRel(1.80 * constants::CODATA08_ANGSTROM_TO_BOHR, 1e-12)
    );
    CHECK_THROWS_AS(config.getGeepSigma(55L), std::runtime_error);
    CHECK_THROWS_AS(config.getGeepSigma(0L), std::out_of_range);
    CHECK_THROWS_AS(config.getGeepSigma(119L), std::out_of_range);

    writeTmpFile(
        fname,
        "&mlmm mlmm_coupling = 'coulombic-geep', geep_sigmas = 'builtin-carbon',\n"
        "  geep_factor = 2.0 /\n"
    );
    config = GlobalConfig::fromNamelistFile(path.string(), "mlmm", "sander");
    for (long atomic_num : {1L, 6L, 54L}) {
        CHECK_THAT(
            config.getGeepSigma(atomic_num),
            WithinRel(3.0 * constants::CODATA08_ANGSTROM_TO_BOHR, 1e-12)
        );
    }
    CHECK_THROWS_AS(config.getGeepSigma(55L), std::runtime_error);

    writeTmpFile(
        fname,
        "&mlmm mlmm_coupling = 'simple-polarizable-geep', use_mlip_atomic_charges = T,\n"
        "  geep_sigmas = 2.0, , 3.0, geep_factor = 0.5 /\n"
    );
    config = GlobalConfig::fromNamelistFile(path.string(), "mlmm", "sander");
    CHECK(config.getGeepSigma(1L) == 1.0);
    CHECK(config.getGeepSigma(3L) == 1.5);
    CHECK_THROWS_AS(config.getGeepSigma(2L), std::runtime_error);
    CHECK_THROWS_AS(config.getGeepSigma(4L), std::runtime_error);
    std::filesystem::remove(path);
}

TEST_CASE("GlobalConfig GEEP validation", "[config]") {
    const std::string fname = "torchani_test_config_geep_validation.in";
    const std::vector<std::string> invalid_texts{
        "&mlmm mlmm_coupling = 'coulombic', geep_sigmas = 1.0 /\n",
        "&mlmm mlmm_coupling = 'coulombic', geep_factor = 1.2 /\n",
        "&mlmm mlmm_coupling = 'coulombic-geep', geep_factor = 0.0 /\n",
        "&mlmm mlmm_coupling = 'coulombic-geep', geep_sigmas = -1.0 /\n",
        "&mlmm mlmm_coupling = 'coulombic-geep', geep_sigmas = 'builtin-nonexistent' /\n",
        "&mlmm mlmm_coupling = 'coulombic-geep', geep_sigmas = 1.0, 'builtin' /\n",
        "&mlmm mlmm_coupling = 'coulombic-geep', use_mlip_atomic_charges = T /\n",
        "&mlmm mlmm_coupling = 'coulombic-geep', inv_pol_dielectric = 1.0 /\n",
        "&mlmm mlmm_coupling = 'coulombic-geep', use_mlip_atomic_volumes = T /\n",
        "&mlmm mlmm_coupling = 'simple-polarizable-geep' /\n",
    };
    for (const std::string& text : invalid_texts) {
        std::filesystem::path path = writeTmpFile(fname, text);
        INFO(text);
        CHECK_THROWS_AS(
            GlobalConfig::fromNamelistFile(path.string(), "mlmm", "sander"),
            config::config_validation_error
        );
        std::filesystem::remove(path);
    }

    // Only supported with the &mlmm namelist
    for (
        const std::string& mlmm_coupling : {"coulombic-geep", "simple-polarizable-geep"}
    ) {
        std::filesystem::path path = writeTmpFile(
            fname,
            "&mlip mlmm_coupling = '" + mlmm_coupling + "',"
            " use_mlip_atomic_charges = T, allow_untested_protocols = T /\n"
        );
        INFO(mlmm_coupling);
        CHECK_THROWS_AS(
            GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip"),
            config::config_validation_error
        );
        std::filesystem::remove(path);
    }

    std::filesystem::path path = writeTmpFile(
        fname,
        "&mlmm mlmm_coupling = 'Simple-Polarizable-GEEP', use_mlip_atomic_charges = T,\n"
        "  use_mlip_atomic_volumes = T, inv_pol_dielectric = 0.5 /\n"
    );
    GlobalConfig config =
        GlobalConfig::fromNamelistFile(path.string(), "mlmm", "sander");
    CHECK(config.getString("mlmm_coupling") == "simple-polarizable-geep");
    CHECK(config.getString("electrostatics_kind") == "hard-cutoff");
    std::string info = capturedPrintInfo(config);
    CHECK(info.find("geep_sigmas") != std::string::npos);
    CHECK(info.find("builtin") != std::string::npos);
    CHECK(info.find("geep_factor") != std::string::npos);
    CHECK(info.find("simple polarizable approximation") != std::string::npos);
    CHECK(info.find("Gaussians (GEEP)") != std::string::npos);

    writeTmpFile(fname, "&mlmm mlmm_coupling = 'coulombic' /\n");
    info = capturedPrintInfo(
        GlobalConfig::fromNamelistFile(path.string(), "mlmm", "sander")
    );
    CHECK(info.find("geep_sigmas") == std::string::npos);
    CHECK(info.find("GEEP") == std::string::npos);
    std::filesystem::remove(path);
}

TEST_CASE("GlobalConfig charge and spin from &qmmm", "[config]") {
    const std::string fname = "torchani_test_config_charge_spin.in";
    // Only in &qmmm: copied to &mlip
    std::filesystem::path path =
        writeTmpFile(fname, "&mlip /\n&qmmm qmcharge = -1, spin = 2 /\n");
    GlobalConfig config =
        GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip", "sander");
    CHECK(config.getInt("ml_charge") == -1);
    CHECK(config.getInt("ml_spin_mult") == 2);
    // &qmmm is only read if the calling program is sander
    config = GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip");
    CHECK(config.getInt("ml_charge") == 0);
    CHECK(config.getInt("ml_spin_mult") == 1);
    // Only in &mlip: kept, but a nonzero charge must also be set in &qmmm
    writeTmpFile(fname, "&mlip ml_charge = 0, ml_spin_mult = 3 /\n&qmmm /\n");
    config = GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip", "sander");
    CHECK(config.getInt("ml_charge") == 0);
    CHECK(config.getInt("ml_spin_mult") == 3);
    writeTmpFile(fname, "&mlip ml_charge = 1 /\n&qmmm /\n");
    CHECK_THROWS_WITH(
        GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip", "sander"),
        Catch::Matchers::ContainsSubstring(
            "ml_charge = 1 in &mlip requires the same qmcharge"
        )
    );
    // In both, consistent
    writeTmpFile(
        fname,
        "&mlip ml_charge = 1, ml_spin_mult = 3 /\n&qmmm qmcharge = 1, spin = 3 /\n"
    );
    CHECK_NOTHROW(
        GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip", "sander")
    );
    // In both, inconsistent
    writeTmpFile(fname, "&mlip ml_charge = 1 /\n&qmmm qmcharge = 0 /\n");
    CHECK_THROWS_WITH(
        GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip", "sander"),
        Catch::Matchers::ContainsSubstring(
            "ml_charge = 1 in &mlip doesn't match qmcharge = 0"
        )
    );
    writeTmpFile(fname, "&mlip ml_spin_mult = 1 /\n&qmmm spin = 2 /\n");
    CHECK_THROWS_WITH(
        GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip", "sander"),
        Catch::Matchers::ContainsSubstring(
            "ml_spin_mult = 1 in &mlip doesn't match spin = 2"
        )
    );
    std::filesystem::remove(path);
}

TEST_CASE("GlobalConfig calling program", "[config]") {
    const std::string fname = "torchani_test_config_calling_program.in";
    std::filesystem::path path =
        writeTmpFile(fname, "&mlip /\n&qmmm qmcharge = -1 /\n");
    // Case-insensitive, and &qmmm is only read for sander
    CHECK(
        GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip", "SANDER")
            .getInt("ml_charge")
        == -1
    );
    CHECK(
        GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip", "pmemd")
            .getInt("ml_charge")
        == 0
    );
    CHECK(
        GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip")
            .getInt("ml_charge")
        == 0
    );
    CHECK_THROWS_WITH(
        GlobalConfig::fromNamelistFile(path.string(), "qmmm-and-mlip", "gromacs"),
        Catch::Matchers::ContainsSubstring("calling_program must be")
    );
    std::filesystem::remove(path);
}
