#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "config.h"
#include "fortran_utils.h"

// c10 defines a CHECK macro, which must not replace the one from Catch2
#undef CHECK
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

using config::GlobalConfig;

static std::filesystem::path writeTmpFile(const std::string& name, const std::string& text) {
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
        "torchani_test_config_defaults.in",
        "title\n&cntrl imin=0 /\n&mlip /\n"
    );
    GlobalConfig config = GlobalConfig::fromNamelistFile(path.string());
    bool use_cuda_device = config::defaultMlipConfig().getBool("use_cuda_device");
    CHECK(config.device() == torch::Device(use_cuda_device ? torch::kCUDA : torch::kCPU, 0));
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
    CHECK(config.getString("no_eval_mask").empty());
    std::filesystem::remove(path);
}

TEST_CASE("GlobalConfig CUDA device", "[config]") {
    std::filesystem::path path = writeTmpFile(
        "torchani_test_config_cuda.in",
        "&mlip use_cuda_device = T, cuda_device_index = 1 /\n"
    );
#ifdef TORCHANI_BUILD_CUDA
    CHECK(GlobalConfig::fromNamelistFile(path.string()).device() == torch::Device(torch::kCUDA, 1));
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
        "  ml_spin_mult = 2, extra_features_file_path = 'features.txt',\n"
        "/\n"
        "&mlip model_type = 'ignored' /\n"
    );
    GlobalConfig config = GlobalConfig::fromNamelistFile(path.string());
    CHECK(config.getInt("cuda_device_index") == 1);
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

    path = writeTmpFile("torchani_test_config_values.in", "&mlip use_cuda_device = F /\n");
    CHECK(GlobalConfig::fromNamelistFile(path.string()).device() == torch::Device(torch::kCPU, 0));

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
    std::filesystem::path path = writeTmpFile("torchani_test_config_errors.in", "&cntrl /\n");
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
    writeTmpFile("torchani_test_config_errors.in", "&mlip net_charge = 1, ml_charge = 1 /\n");
    CHECK_THROWS_AS(
        GlobalConfig::fromNamelistFile(path.string()),
        fortran_utils::namelist_alias_conflict_error
    );
    writeTmpFile(
        "torchani_test_config_errors.in", "&mlip model_type = 'nutmeg-small::features.txt' /\n"
    );
    CHECK_THROWS_AS(
        GlobalConfig::fromNamelistFile(path.string()),
        config::config_validation_error
    );
    writeTmpFile("torchani_test_config_errors.in", "&mlip /\n&ani /\n");
    CHECK_THROWS_AS(GlobalConfig::fromNamelistFile(path.string()), std::runtime_error);
    std::filesystem::remove(path);
}

TEST_CASE("GlobalConfig from deprecated ani namelist", "[config]") {
    std::filesystem::path path = writeTmpFile(
        "torchani_test_config_ani.in",
        "&cntrl /\n&ani model_type = 'animbis' /\n"
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
    GlobalConfig config = GlobalConfig::fromNamelistFile(path.string(), true, "sander");
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
    CHECK(config.getDouble("pol_h") == 2.5);
    CHECK(config.getDouble("pol_c") == 11.3);
    CHECK(config.getDouble("pol_ar") == 10.0);
    CHECK(config.getDouble("pol_fe") == 50.0);
    CHECK(std::isnan(config.getDouble("pol_og")));
    CHECK(config.getBool("write_mlmm_coords"));
    CHECK_FALSE(config.getBool("write_mlmm_forces"));
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
    config = GlobalConfig::fromNamelistFile(path.string(), true);
    CHECK(config.getBool("use_qm_coupling"));
    CHECK(config.getString("qm_coupling_method") == "orca");
    writeTmpFile("torchani_test_config_mlmm.in", "&mlip /\n");
    CHECK_THROWS_AS(GlobalConfig::fromNamelistFile(path.string()).mlmm(), std::logic_error);
    config = GlobalConfig::fromNamelistFile(path.string(), true);
    CHECK(config.getString("mlmm_coupling") == "coulombic");

    writeTmpFile(
        "torchani_test_config_mlmm.in",
        "&mlip mlmm_coupling = 'Simple-Polarizable', use_mlip_atomic_charges = T,\n"
        "  write_charges = T, write_charges_grad = T, write_forces = T,\n"
        "  use_hybrid_mlqm = T, hybrid_mlqm_method = 'gau', hybrid_mlqm_switch_high = 0.5 /\n"
    );
    config = GlobalConfig::fromNamelistFile(path.string(), true);
    CHECK(config.getString("mlmm_coupling") == "simple-polarizable");
    CHECK(config.getBool("use_mlip_atomic_charges"));
    CHECK(config.getBool("write_mlmm_atomic_charges"));
    CHECK_FALSE(config.getBool("write_mlmm_coords"));
    CHECK(config.getBool("write_mlmm_forces"));
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
        "&mlip mlmm_coupling = 'custom', write_mlmm_atomic_charges = T /\n",
        "&mlip write_mlmm_atomic_charges = T /\n",
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
            GlobalConfig::fromNamelistFile(path.string(), true),
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
    GlobalConfig::fromNamelistFile(path.string(), true);
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
    GlobalConfig::fromNamelistFile(path.string(), true);
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
    CHECK_THROWS_AS(GlobalConfig::fromNamelistFile(path.string(), true, "sander"), std::runtime_error);
    CHECK_NOTHROW(GlobalConfig::fromNamelistFile(path.string(), true));
    writeTmpFile(fname, "&mlip /\n&qmmm qmmm_int = 2 /\n");
    CHECK_NOTHROW(GlobalConfig::fromNamelistFile(path.string(), true));
    writeTmpFile(
        fname,
        "&mlip use_qm_coupling = T, qm_coupling_method = 'orca', mlmm_coupling = 'none' /\n"
        "&qmmm qmcharge = 1 /\n"
    );
    CHECK_NOTHROW(GlobalConfig::fromNamelistFile(path.string(), true, "sander"));
    writeTmpFile(
        fname,
        "&mlip use_qm_coupling = T, qm_coupling_method = 'orca', use_mlip_atomic_charges = T,\n"
        "  mlmm_coupling = 'none', allow_untested_protocols = T /\n&qmmm qmcharge = 1 /\n"
    );
    std::stringstream captured_charges;
    std::streambuf* charges_cerr_buf = std::cerr.rdbuf(captured_charges.rdbuf());
    CHECK_NOTHROW(GlobalConfig::fromNamelistFile(path.string(), true, "sander"));
    std::cerr.rdbuf(charges_cerr_buf);
    CHECK(
        captured_charges.str().find("use_mlip_atomic_charges has no effect") != std::string::npos
    );
    writeTmpFile(fname, "&mlip /\n&qmmm qmcharge = 'a' /\n");
    CHECK_THROWS_AS(
        GlobalConfig::fromNamelistFile(path.string(), true, "sander"),
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
            GlobalConfig::fromNamelistFile(path.string(), true, "sander"),
            config::config_validation_error
        );
    }

    writeTmpFile(
        fname, "&mlip mlmm_coupling = 'none', allow_untested_protocols = T /\n&qmmm qmmm_int = 0 /\n"
    );
    std::stringstream captured;
    std::streambuf* cerr_buf = std::cerr.rdbuf(captured.rdbuf());
    GlobalConfig::fromNamelistFile(path.string(), true, "sander");
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
    std::filesystem::path path = writeTmpFile(fname, "&mlip model_type = 'animbis' /\n");
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
    info = capturedPrintInfo(GlobalConfig::fromNamelistFile(path.string(), true, "sander"));
    CHECK(info.find("write_mlmm_coords") != std::string::npos);
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
    info = capturedPrintInfo(GlobalConfig::fromNamelistFile(path.string(), true, "sander"));
    std::cerr.rdbuf(cerr_buf);
    CHECK(info.find("INTERNAL OPTIONS") != std::string::npos);
    CHECK(info.find("hybrid_mlqm_switch_low") != std::string::npos);
    CHECK(info.find("mlmm_coupling") == std::string::npos);
    CHECK(info.find("calculated by MLIP Interface") == std::string::npos);

    writeTmpFile(fname, "&mlip mlmm_coupling = 'none' /\n&qmmm /\n");
    info = capturedPrintInfo(GlobalConfig::fromNamelistFile(path.string(), true, "sander"));
    CHECK(info.find("only the in vacuo ML energy") != std::string::npos);
    CHECK(info.find("calculated by MLIP Interface") == std::string::npos);
    std::filesystem::remove(path);
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

TEST_CASE("GlobalConfig charge and spin from &qmmm", "[config]") {
    const std::string fname = "torchani_test_config_charge_spin.in";
    // Only in &qmmm: copied to &mlip
    std::filesystem::path path =
        writeTmpFile(fname, "&mlip /\n&qmmm qmcharge = -1, spin = 2 /\n");
    GlobalConfig config = GlobalConfig::fromNamelistFile(path.string(), true, "sander");
    CHECK(config.getInt("ml_charge") == -1);
    CHECK(config.getInt("ml_spin_mult") == 2);
    // &qmmm is only read if the calling program is sander
    config = GlobalConfig::fromNamelistFile(path.string(), true);
    CHECK(config.getInt("ml_charge") == 0);
    CHECK(config.getInt("ml_spin_mult") == 1);
    // Only in &mlip: kept, but a nonzero charge must also be set in &qmmm
    writeTmpFile(fname, "&mlip ml_charge = 0, ml_spin_mult = 3 /\n&qmmm /\n");
    config = GlobalConfig::fromNamelistFile(path.string(), true, "sander");
    CHECK(config.getInt("ml_charge") == 0);
    CHECK(config.getInt("ml_spin_mult") == 3);
    writeTmpFile(fname, "&mlip ml_charge = 1 /\n&qmmm /\n");
    CHECK_THROWS_WITH(
        GlobalConfig::fromNamelistFile(path.string(), true, "sander"),
        Catch::Matchers::ContainsSubstring("ml_charge = 1 in &mlip requires the same qmcharge")
    );
    // In both, consistent
    writeTmpFile(fname, "&mlip ml_charge = 1, ml_spin_mult = 3 /\n&qmmm qmcharge = 1, spin = 3 /\n");
    CHECK_NOTHROW(GlobalConfig::fromNamelistFile(path.string(), true, "sander"));
    // In both, inconsistent
    writeTmpFile(fname, "&mlip ml_charge = 1 /\n&qmmm qmcharge = 0 /\n");
    CHECK_THROWS_WITH(
        GlobalConfig::fromNamelistFile(path.string(), true, "sander"),
        Catch::Matchers::ContainsSubstring("ml_charge = 1 in &mlip doesn't match qmcharge = 0")
    );
    writeTmpFile(fname, "&mlip ml_spin_mult = 1 /\n&qmmm spin = 2 /\n");
    CHECK_THROWS_WITH(
        GlobalConfig::fromNamelistFile(path.string(), true, "sander"),
        Catch::Matchers::ContainsSubstring("ml_spin_mult = 1 in &mlip doesn't match spin = 2")
    );
    std::filesystem::remove(path);
}

TEST_CASE("GlobalConfig calling program", "[config]") {
    const std::string fname = "torchani_test_config_calling_program.in";
    std::filesystem::path path =
        writeTmpFile(fname, "&mlip /\n&qmmm qmcharge = -1 /\n");
    // Case-insensitive, and &qmmm is only read for sander
    CHECK(GlobalConfig::fromNamelistFile(path.string(), true, "SANDER").getInt("ml_charge") == -1);
    CHECK(GlobalConfig::fromNamelistFile(path.string(), true, "pmemd").getInt("ml_charge") == 0);
    CHECK(GlobalConfig::fromNamelistFile(path.string(), true).getInt("ml_charge") == 0);
    CHECK_THROWS_WITH(
        GlobalConfig::fromNamelistFile(path.string(), true, "gromacs"),
        Catch::Matchers::ContainsSubstring("calling_program must be")
    );
    std::filesystem::remove(path);
}
