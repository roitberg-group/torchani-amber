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

using config::GlobalConfig;

static std::filesystem::path writeTmpFile(const std::string& name, const std::string& text) {
    std::filesystem::path path = std::filesystem::temp_directory_path() / name;
    std::ofstream file(path);
    file << text;
    return path;
}

TEST_CASE("GlobalConfig getters throw before the config is read", "[config]") {
    GlobalConfig config{};
    CHECK_THROWS_AS(config.mlipConfig(), std::logic_error);
    CHECK_THROWS_AS(config.mlmmConfig(), std::logic_error);
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
    CHECK(config.mlipConfig().getBool("use_cuaev"));
    CHECK(config.mlipConfig().getString("model_type") == "ani2x");
    CHECK_FALSE(config.mlipConfig().getBool("use_external_nonbond"));
    CHECK(config.mlipConfig().getInt("model_index") == -1);
    CHECK_FALSE(config.mlipConfig().getBool("use_external_neighborlist"));
    CHECK(config.mlipConfig().getInt("ml_charge") == 0);
    CHECK(config.mlipConfig().getInt("ml_spin_mult") == 1);
    CHECK(config.mlipConfig().getString("no_eval_mask").empty());
    std::filesystem::remove(path);
}

TEST_CASE("GlobalConfig from namelist file values", "[config]") {
    std::filesystem::path path = writeTmpFile(
        "torchani_test_config_values.in",
        "&mlip\n"
        "  model_type = 'animbis', use_cuda_device = .true., use_double_precision = F,\n"
        "  net_charge = -1, no_eval_mask = ':WAT', use_amber_neighborlist = T,\n"
        "  use_all_amber_nonbond = T, use_cuaev = F, model_index = 3, cuda_device_index = 1,\n"
        "  ml_spin_mult = 2,\n"
        "/\n"
        "&mlip model_type = 'ignored' /\n"
    );
    GlobalConfig config = GlobalConfig::fromNamelistFile(path.string());
    CHECK(config.device() == torch::Device(torch::kCUDA, 1));
    CHECK(config.dtype() == torch::kFloat32);
    CHECK(config.useNoEvalAtoms());
    CHECK_FALSE(config.mlipConfig().getBool("use_cuaev"));
    CHECK(config.mlipConfig().getString("model_type") == "animbis");
    CHECK(config.mlipConfig().getBool("use_external_nonbond"));
    CHECK(config.mlipConfig().getInt("model_index") == 3);
    CHECK(config.mlipConfig().getBool("use_external_neighborlist"));
    CHECK(config.mlipConfig().getInt("ml_charge") == -1);
    CHECK(config.mlipConfig().getInt("ml_spin_mult") == 2);
    CHECK(config.mlipConfig().getString("no_eval_mask") == ":WAT");
    std::filesystem::remove(path);

    path = writeTmpFile("torchani_test_config_values.in", "&mlip use_cuda_device = F /\n");
    CHECK(GlobalConfig::fromNamelistFile(path.string()).device() == torch::Device(torch::kCPU, 0));

    writeTmpFile(
        "torchani_test_config_values.in",
        "&mlip ml_charge = 2, use_external_neighborlist = T /\n"
    );
    config = GlobalConfig::fromNamelistFile(path.string());
    CHECK(config.mlipConfig().getInt("ml_charge") == 2);
    CHECK(config.mlipConfig().getBool("use_external_neighborlist"));
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
    CHECK(config.mlipConfig().getString("model_type") == "animbis");
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
        "  qlow = 0.1, use_charges_derivatives = F, use_extcoupling = T, use_torch_coupling = F,\n"
        "  extcoupling_program = 'orca' /\n"
        "&qmmm qmmask = ':1', qm_theory = 'EXTERN', qmcharge = 0, qm_ewald = 0, spin = 2,\n"
        "  qmcut = 8.0 /\n"
    );
    GlobalConfig config = GlobalConfig::fromNamelistFile(path.string(), true, true);
    const config::ConfigDict& mlmm = config.mlmmConfig();
    CHECK(mlmm.getInt("mlmm_coupling") == 1);
    CHECK(mlmm.getBool("use_mlip_atomic_charges"));
    CHECK_FALSE(mlmm.getBool("use_mlip_atomic_charges_grad"));
    CHECK_FALSE(mlmm.getBool("use_numerical_mlmm_forces"));
    CHECK_FALSE(mlmm.getBool("use_interface_coupling"));
    CHECK_FALSE(mlmm.getBool("allow_untested_protocols"));
    CHECK(mlmm.getDouble("inv_pol_dielectric") == 1.0);
    CHECK(mlmm.getDouble("pol_h") == 2.5);
    CHECK(mlmm.getDouble("pol_c") == 11.3);
    CHECK(mlmm.getDouble("pol_ar") == 10.0);
    CHECK(mlmm.getDouble("pol_fe") == 50.0);
    CHECK(std::isnan(mlmm.getDouble("pol_og")));
    CHECK(mlmm.getBool("write_ml_coords"));
    CHECK_FALSE(mlmm.getBool("write_ml_forces"));
    CHECK(mlmm.getBool("use_hybrid_mlqm"));
    CHECK(mlmm.getString("hybrid_mlqm_method") == "orca");
    CHECK(mlmm.getDouble("hybrid_mlqm_switch_low") == 0.1);
    CHECK(mlmm.getDouble("hybrid_mlqm_switch_high") == 0.3);
    CHECK(mlmm.getBool("use_qm_coupling"));
    CHECK(mlmm.getString("qm_coupling_method") == "orca");

    CHECK_THROWS_AS(
        GlobalConfig::fromNamelistFile(path.string()),
        fortran_utils::namelist_unknown_key_error
    );
    writeTmpFile("torchani_test_config_mlmm.in", "&mlip /\n");
    CHECK_THROWS_AS(GlobalConfig::fromNamelistFile(path.string()).mlmmConfig(), std::logic_error);
    config = GlobalConfig::fromNamelistFile(path.string(), true);
    CHECK(config.mlmmConfig().getInt("mlmm_coupling") == 0);

    writeTmpFile(
        "torchani_test_config_mlmm.in",
        "&mlip mlmm_coupling = 1, use_mlip_atomic_charges = T, write_charges = T,\n"
        "  write_charges_grad = T, write_forces = T, use_interface_coupling = F,\n"
        "  use_hybrid_mlqm = T, hybrid_mlqm_method = 'gau', hybrid_mlqm_switch_high = 0.5 /\n"
    );
    config = GlobalConfig::fromNamelistFile(path.string(), true);
    CHECK(config.mlmmConfig().getBool("use_mlip_atomic_charges"));
    CHECK(config.mlmmConfig().getBool("write_ml_atomic_charges"));
    CHECK(config.mlmmConfig().getBool("write_ml_atomic_charges_grad"));
    CHECK_FALSE(config.mlmmConfig().getBool("write_ml_coords"));
    CHECK(config.mlmmConfig().getBool("write_ml_forces"));
    CHECK_FALSE(config.mlmmConfig().getBool("use_interface_coupling"));
    CHECK(config.mlmmConfig().getBool("use_hybrid_mlqm"));
    CHECK(config.mlmmConfig().getString("hybrid_mlqm_method") == "gau");
    CHECK(config.mlmmConfig().getDouble("hybrid_mlqm_switch_high") == 0.5);
    CHECK(config.mlmmConfig().getString("qm_coupling_method") == "none");
    std::filesystem::remove(path);
}

TEST_CASE("GlobalConfig ML/MM config validation", "[config]") {
    const std::string fname = "torchani_test_config_validation.in";
    const std::vector<std::string> invalid_texts{
        "&mlip use_external_neighborlist = T /\n",
        "&mlip ml_charge = 1, use_interface_coupling = F /\n",
        "&mlip use_hybrid_mlqm = T, hybrid_mlqm_method = 'orca' /\n",
        "&mlip mlmm_coupling = 3 /\n",
        "&mlip mlmm_coupling = 0, use_mlip_atomic_charges = T /\n",
        "&mlip mlmm_coupling = 1 /\n",
        "&mlip inv_pol_dielectric = 1.0 /\n",
        "&mlip use_qm_coupling = T, qm_coupling_method = 'orca' /\n",
        "&mlip mlmm_coupling = 2, use_qm_coupling = T, qm_coupling_method = 'orca',\n"
        "  use_interface_coupling = F /\n",
        "&mlip mlmm_coupling = 2, use_mlip_atomic_charges = T, allow_untested_protocols = T /\n",
        "&mlip mlmm_coupling = 2, write_ml_atomic_charges = T /\n",
        "&mlip use_qm_coupling = T, use_interface_coupling = F /\n",
        "&mlip use_hybrid_mlqm = T, use_interface_coupling = F /\n",
        "&mlip qm_coupling_method = 'orca' /\n",
        "&mlip hybrid_mlqm_method = 'orca' /\n",
        "&mlip use_numerical_mlmm_forces = T /\n",
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
        "&mlip mlmm_coupling = 0, use_mlip_atomic_charges = T, allow_untested_protocols = T /\n"
    );
    std::stringstream captured;
    std::streambuf* cerr_buf = std::cerr.rdbuf(captured.rdbuf());
    GlobalConfig::fromNamelistFile(path.string(), true);
    std::cerr.rdbuf(cerr_buf);
    CHECK(captured.str().find("untested protocol") != std::string::npos);
    std::filesystem::remove(path);
}

TEST_CASE("GlobalConfig Amber &qmmm namelist", "[config]") {
    const std::string fname = "torchani_test_config_qmmm.in";
    std::filesystem::path path = writeTmpFile(fname, "&mlip /\n");
    CHECK_THROWS_AS(GlobalConfig::fromNamelistFile(path.string(), true, true), std::runtime_error);
    CHECK_NOTHROW(GlobalConfig::fromNamelistFile(path.string(), true));
    writeTmpFile(fname, "&mlip /\n&qmmm qmmm_int = 2 /\n");
    CHECK_NOTHROW(GlobalConfig::fromNamelistFile(path.string(), true));
    writeTmpFile(fname, "&mlip /\n&qmmm qmcharge = 'a' /\n");
    CHECK_THROWS_AS(
        GlobalConfig::fromNamelistFile(path.string(), true, true),
        fortran_utils::namelist_type_error
    );

    const std::vector<std::string> invalid_texts{
        "&mlip use_interface_coupling = F /\n&qmmm qmcharge = 1 /\n",
        "&mlip allow_untested_protocols = T /\n&qmmm qmmm_int = 2 /\n",
        "&mlip /\n&qmmm qmmm_int = 0 /\n",
        "&mlip mlmm_coupling = 1, use_mlip_atomic_charges = T, allow_untested_protocols = T /\n"
        "&qmmm qmmm_int = 0 /\n",
        "&mlip allow_untested_protocols = T /\n&qmmm qmmm_int = 5 /\n",
    };
    for (const std::string& text : invalid_texts) {
        writeTmpFile(fname, text);
        CHECK_THROWS_AS(
            GlobalConfig::fromNamelistFile(path.string(), true, true),
            config::config_validation_error
        );
    }

    writeTmpFile(fname, "&mlip allow_untested_protocols = T /\n&qmmm qmmm_int = 0 /\n");
    std::stringstream captured;
    std::streambuf* cerr_buf = std::cerr.rdbuf(captured.rdbuf());
    GlobalConfig::fromNamelistFile(path.string(), true, true);
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

    writeTmpFile(
        fname,
        "&mlip mlmm_coupling = 1, use_mlip_atomic_charges = T, write_xyz = T /\n&qmmm /\n"
    );
    info = capturedPrintInfo(GlobalConfig::fromNamelistFile(path.string(), true, true));
    CHECK(info.find("write_ml_coords") != std::string::npos);
    CHECK(info.find("write_xyz") == std::string::npos);
    CHECK(info.find("(simple polarizable)") != std::string::npos);
    CHECK(info.find("calculated by MLIP Interface") != std::string::npos);
    CHECK(info.find("NN-predicted charges") != std::string::npos);
    CHECK(info.find("INTERNAL OPTIONS") == std::string::npos);

    writeTmpFile(
        fname,
        "&mlip allow_untested_protocols = T, use_mlip_atomic_charges_grad = F /\n"
        "&qmmm qmmm_int = 0 /\n"
    );
    std::stringstream captured_cerr;
    std::streambuf* cerr_buf = std::cerr.rdbuf(captured_cerr.rdbuf());
    info = capturedPrintInfo(GlobalConfig::fromNamelistFile(path.string(), true, true));
    std::cerr.rdbuf(cerr_buf);
    CHECK(info.find("INTERNAL OPTIONS") != std::string::npos);
    CHECK(info.find("use_mlip_atomic_charges_grad") != std::string::npos);
    CHECK(info.find("mlmm_coupling") == std::string::npos);
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
    CHECK_THROWS_AS(config.mlmmConfig(), std::logic_error);

    mlip_config = mlip_config.withValue("no_eval_mask", std::string{":WAT"});
    CHECK(GlobalConfig{mlip_config}.useNoEvalAtoms());
}

TEST_CASE("Config dict getters", "[config]") {
    config::ConfigDict mlip_config = config::defaultMlipConfig();
    CHECK(mlip_config.getInt("ml_spin_mult") == 1);
    CHECK_THROWS_AS(mlip_config.getBool("ml_spin_mult"), std::bad_variant_access);
    CHECK_THROWS_AS(mlip_config.getInt("no_such_key"), std::out_of_range);
    mlip_config = mlip_config.withValue("ml_charge", 1L << 40);
    CHECK_THROWS_AS(mlip_config.getInt("ml_charge"), std::out_of_range);
}
