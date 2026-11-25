#include <torchani.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <unistd.h>

#include <torch/all.h>

#include "constants.h"
#include "mlip.h"
#include "tensor_utils.h"

// c10 defines a CHECK macro, which must not replace the one from Catch2
#undef CHECK
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

// Default tolerances are:
// 4.5e-6 rel, 3.5e-5 abs for float32 (x3 than torch, needed for ensembles)
// 1e-7 rel, 1e-7 abs for float64 (same as torch)

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

#define FLOAT_EQ(pred, expect, dbl) \
    CHECK_THAT( \
        pred, \
        WithinAbs(expect, dbl ? 1e-7 : 4.5e-6) \
            || WithinRel(expect, dbl ? 1e-7 : 3.5e-5) \
    )



// Write an &mlip namelist with the given options to a temporary file, return its path
static std::string writeMlipConfig(
    const std::string& model_type,
    int model_index,
    bool use_double_precision,
    bool use_cuda_device,
    bool use_cuaev,
    int cuda_device_index = 0,
    const std::string& extra_options = ""
) {
    std::filesystem::path path =
        std::filesystem::temp_directory_path()
        / ("torchani_test_mlip_config_" + std::to_string(getpid()) + ".in");
    std::ofstream file(path);
    file << "&mlip\n"
         << "  model_type = '" << model_type << "',\n"
         << "  model_index = " << model_index << ",\n"
         << "  use_double_precision = " << (use_double_precision ? "T" : "F") << ",\n"
         << "  use_cuda_device = " << (use_cuda_device ? "T" : "F") << ",\n"
         << "  use_cuaev = " << (use_cuaev ? "T" : "F") << ",\n"
         << "  cuda_device_index = " << cuda_device_index << ",\n"
         << extra_options
         << "/\n";
    return path.string();
}

// Derivatives of the atomic charges wrt the coords, calculated with the internal model
// on the CPU, flattened as dq[c + 3 * j + 3 * num_atoms * k] = d(charge k) / d(coord c
// of atom j)
static std::vector<double> calcAtomicChargeGrads(
    const std::string& model_path,
    int num_atoms,
    int atomic_numbers[],
    double coords[][3],
    int net_charge,
    bool use_double
) {
    torch::ScalarType dtype = use_double ? torch::kFloat64 : torch::kFloat32;
    std::unique_ptr<mlip::MlipModel> model =
        mlip::MlipModel::from_model_path(model_path, torch::kCPU, dtype);
    torch::Tensor atomic_numbers_tensor =
        torch::from_blob(atomic_numbers, {1, num_atoms}, torch::kInt32).to(torch::kLong);
    torch::Tensor coords_tensor = torch::from_blob(coords, {1, num_atoms, 3}, torch::kFloat64)
        .to(dtype)
        .requires_grad_(true);
    mlip::MlipOutput output = model->computeSinglePoint(
        atomic_numbers_tensor,
        coords_tensor,
        /*cell=*/std::nullopt,
        net_charge,
        /*spin_mult=*/1,
        /*atomic=*/false,
        /*ensemble_values=*/false,
        /*molecule_idxs=*/std::nullopt,
        /*calc_forces=*/false
    );
    torch::Tensor atomic_charges = output.at("atomic_charges");
    std::vector<torch::Tensor> charge_grads;
    for (int atom = 0; atom != num_atoms; ++atom) {
        torch::Tensor charge_grad = torch::autograd::grad(
            {atomic_charges.index({0, atom})},
            {coords_tensor},
            /*grad_outputs=*/{},
            /*retain_graph=*/true
        )[0];
        charge_grads.push_back(charge_grad.squeeze(0));
    }
    torch::Tensor grads = torch::stack(charge_grads).to(torch::kFloat64).contiguous();
    return std::vector<double>(grads.data_ptr<double>(), grads.data_ptr<double>() + grads.numel());
}

static std::string file_path = __FILE__;
static std::string file_directory = file_path.substr(
    0,
    file_path.find_last_of("/")
) + "/";


TEST_CASE("C bindings") {
    double coords[][3] = {{3.0, 3.0, 4.0}, {1.0, 2.0, 1.0}};
    int molecule_idxs_buf[] = {1, 1};
    int atomic_numbers[] = {1, 6};
    int no_eval_atoms[] = {0, 0};
    int size = 2;
    double forces[size][3];
    double qbc;
    double qbc_deriv[size][3];
    int charges_type_raw = 0;
    double atomic_charges[size];
    double cell[3][3] = {{10., 0., 0.}, {0., 10., 0.}, {0., 0., 10.}};
    int use_double = GENERATE(0, 1);
    INFO("Comparing with double precision: " << use_double);

    SECTION("Energy + Force", "Test energy and force for all models") {
        using ModelSpec = std::tuple<std::string, int, std::string>;
        auto model_spec = GENERATE(
            table<std::string, int, std::string>(
                {
                    ModelSpec{"ani1x", 0, "_1x"},
                    ModelSpec{"ani1ccx", 0, "_1ccx"},
                    ModelSpec{"ani2x", 0, "_2x"},
                    ModelSpec{"animbis", 0, "_mbis"},
                    ModelSpec{"ani1x", -1, "_1x_ensemble"},
                    ModelSpec{"ani1ccx", -1, "_1ccx_ensemble"},
                    ModelSpec{"ani2x", -1, "_2x_ensemble"},
                    ModelSpec{"animbis", -1, "_mbis_ensemble"}
                }
            )
        );
        std::string model_type = std::get<0>(model_spec);
        int network_index = std::get<1>(model_spec);
        std::string model_suffix = std::get<2>(model_spec);
        INFO("Testing model: " << "ani" << model_suffix);

        using DeviceSpec = std::tuple<int, int, int, std::string>;
        auto device_spec = GENERATE(
            table<int, int, int, std::string>(
                {
                    DeviceSpec{0, 0, 0, "_cpu"},  // CPU
                    DeviceSpec{0, 0, 1, "_cpu"},  // CPU + cuAEV
                    DeviceSpec{1, 0, 0, "_cuda"},  // CUDA
                    DeviceSpec{1, 0, 1, "_cuda"}  // CUDA + cuAEV
                }
            )
        );
        int use_cuda_device = std::get<0>(device_spec);
        if (use_cuda_device and std::getenv("TORCHANI_SKIP_CUDA_TESTS")) {
            SKIP("CUDA tests disabled by TORCHANI_SKIP_CUDA_TESTS");
        }
        int device_index = std::get<1>(device_spec);
        int use_cuaev = std::get<2>(device_spec);
        std::string device_suffix = std::get<3>(device_spec);
        INFO("Testing model on device: " << "device" << device_suffix);
        INFO("cuAEV: " << (use_cuaev == 1));
        // TODO: Cell seems to be too small for pbc, but this should generate
        // tests for the cell too

        // output
        double pot_energy;

        mlip_initialize(
            size,
            atomic_numbers,
            writeMlipConfig(
                model_type, network_index, use_double, use_cuda_device, use_cuaev, device_index,
                model_type == "animbis"
                    ? "  use_mlip_atomic_charges = T, allow_untested_protocols = T,\n"
                    : ""
            ).c_str(),
            true,
            "none",
            false
        );

        std::string file_path = file_directory
            + "test_values"
            + device_suffix
            + model_suffix
            + ".txt";

        std::ifstream infile(file_path);
        if (!infile){
            std::cerr
                << "Can't open file"
                << file_path
                << "with test values."
                << '\n';
            std::exit(2);
        }

        if (model_type != "animbis") {
            double test_values[7];
            for (int j = 0; j != 7; ++j) {
                infile >> test_values[j];
            }
            for (long j = 0; j != 10; ++j) {
                mlip_calc_energy_force(
                    size,
                    coords,
                    cell,
                    true,
                    molecule_idxs_buf,
                    false,
                    forces,
                    &pot_energy
                );
                FLOAT_EQ(pot_energy, test_values[0], use_double);
                FLOAT_EQ(forces[0][0], test_values[1], use_double);
                FLOAT_EQ(forces[0][1], test_values[2], use_double);
                FLOAT_EQ(forces[0][2], test_values[3], use_double);
                FLOAT_EQ(forces[1][0], test_values[4], use_double);
                FLOAT_EQ(forces[1][1], test_values[5], use_double);
                FLOAT_EQ(forces[1][2], test_values[6], use_double);
            }
        } else if (network_index == -1) {
            double test_values[28];
            for (int j = 0; j != 28; ++j) {
                infile >> test_values[j];
            }
            std::vector<double> atomic_charge_derivatives = calcAtomicChargeGrads(
                std::string(TORCHANI_MLIPS_BUILD_DIR) + "/animbis.pt",
                size,
                atomic_numbers,
                coords,
                0,
                use_double
            );
            for (long j = 0; j != 10; ++j){
                mlip_calc_energy_force_mlmm_qbc(
                    size,
                    /* mm_atoms_num= */ 0,
                    coords,
                    /* mm_coords= */ nullptr,
                    /* mm_atomic_charges= */ nullptr,
                    /* cell= */ nullptr,
                    /* use_pbc= */ false,
                    /* verbose= */ false,
                    /* outputs */
                    forces,
                    /* mm_forces= */ nullptr,
                    &pot_energy
                );
                int qbc_grad_dims[2] = {size, 3};
                int charges_dims[1] = {size};
                mlip_fetch_result("qbc", &qbc, nullptr, 0);
                mlip_fetch_result("qbc_grad", &qbc_deriv[0][0], qbc_grad_dims, 2);
                mlip_fetch_result("atomic_charges", atomic_charges, charges_dims, 1);
                FLOAT_EQ(pot_energy, test_values[0], use_double);
                FLOAT_EQ(forces[0][0], test_values[1], use_double);
                FLOAT_EQ(forces[0][1], test_values[2], use_double);
                FLOAT_EQ(forces[0][2], test_values[3], use_double);
                FLOAT_EQ(forces[1][0], test_values[4], use_double);
                FLOAT_EQ(forces[1][1], test_values[5], use_double);
                FLOAT_EQ(forces[1][2], test_values[6], use_double);
                FLOAT_EQ(qbc, test_values[7], use_double);
                FLOAT_EQ(qbc_deriv[0][0], test_values[8], use_double);
                FLOAT_EQ(qbc_deriv[0][1], test_values[9], use_double);
                FLOAT_EQ(qbc_deriv[0][2], test_values[10], use_double);
                FLOAT_EQ(qbc_deriv[1][0], test_values[11], use_double);
                FLOAT_EQ(qbc_deriv[1][1], test_values[12], use_double);
                FLOAT_EQ(qbc_deriv[1][2], test_values[13], use_double);
                FLOAT_EQ(atomic_charges[0], test_values[14], use_double);
                FLOAT_EQ(atomic_charges[1], test_values[15], use_double);
                FLOAT_EQ(atomic_charge_derivatives[0], test_values[16], use_double);
                FLOAT_EQ(atomic_charge_derivatives[1], test_values[17], use_double);
                FLOAT_EQ(atomic_charge_derivatives[2], test_values[18], use_double);
                FLOAT_EQ(atomic_charge_derivatives[3], test_values[19], use_double);
                FLOAT_EQ(atomic_charge_derivatives[4], test_values[20], use_double);
                FLOAT_EQ(atomic_charge_derivatives[5], test_values[21], use_double);
                FLOAT_EQ(atomic_charge_derivatives[6], test_values[22], use_double);
                FLOAT_EQ(atomic_charge_derivatives[7], test_values[23], use_double);
                FLOAT_EQ(atomic_charge_derivatives[8], test_values[24], use_double);
                FLOAT_EQ(atomic_charge_derivatives[9], test_values[25], use_double);
                FLOAT_EQ(atomic_charge_derivatives[10], test_values[26], use_double);
                FLOAT_EQ(atomic_charge_derivatives[11], test_values[27], use_double);
            }
        }
    }
}

static const double COUPLING_INV_POL_DIELECTRIC = 0.8;
static const double COUPLING_ATOMIC_ALPHAS[] = {0.5, 1.2};
static const double COUPLING_MM_ATOMIC_CHARGES[] = {0.5, -0.4};

struct CouplingOutput {
    double ml_forces[2][3];
    double mm_forces[2][3];
    double ml_atomic_charges[2];
    double pot_energy;
};

static CouplingOutput calc_coupling(
    double ml_coords[][3], double mm_coords[][3], bool verbose = false
) {
    double mm_atomic_charges[2] = {
        COUPLING_MM_ATOMIC_CHARGES[0], COUPLING_MM_ATOMIC_CHARGES[1]
    };
    CouplingOutput out{};
    mlip_calc_energy_force_mlmm(
        2,
        2,
        ml_coords,
        mm_coords,
        mm_atomic_charges,
        /* cell= */ nullptr,
        /* use_pbc= */ false,
        verbose,
        out.ml_forces,
        out.mm_forces,
        &out.pot_energy
    );
    int charges_dims[1] = {2};
    mlip_fetch_result("atomic_charges", out.ml_atomic_charges, charges_dims, 1);
    return out;
}

TEST_CASE("C bindings for other entry points") {
    const int size = 2;
    double coords[size][3] = {{3.0, 3.0, 4.0}, {1.0, 2.0, 1.0}};
    int molecule_idxs_buf[size] = {1, 1};
    int atomic_numbers[size] = {1, 6};
    int no_eval_atoms[size] = {0, 0};
    double forces[size][3];
    double pot_energy;
    int use_double = GENERATE(0, 1);
    INFO("Comparing with double precision: " << use_double);

    using ModelSpec = std::tuple<std::string, int, std::string>;
    auto model_spec = GENERATE(
        table<std::string, int, std::string>(
            {
                ModelSpec{"ani1x", 0, "_1x"},
                ModelSpec{"ani1ccx", 0, "_1ccx"},
                ModelSpec{"ani2x", 0, "_2x"},
                ModelSpec{"animbis", 0, "_mbis"},
                ModelSpec{"ani1x", -1, "_1x_ensemble"},
                ModelSpec{"ani1ccx", -1, "_1ccx_ensemble"},
                ModelSpec{"ani2x", -1, "_2x_ensemble"},
                ModelSpec{"animbis", -1, "_mbis_ensemble"}
            }
        )
    );
    std::string model_type = std::get<0>(model_spec);
    int network_index = std::get<1>(model_spec);
    std::string model_suffix = std::get<2>(model_spec);
    INFO("Testing model: " << "ani" << model_suffix);

    using DeviceSpec = std::tuple<int, int, std::string>;
    auto device_spec = GENERATE(
        table<int, int, std::string>(
            {
                DeviceSpec{0, 0, "_cpu"},
                DeviceSpec{1, 0, "_cuda"},
                DeviceSpec{1, 1, "_cuda"}
            }
        )
    );
    int use_cuda_device = std::get<0>(device_spec);
    if (use_cuda_device and std::getenv("TORCHANI_SKIP_CUDA_TESTS")) {
        SKIP("CUDA tests disabled by TORCHANI_SKIP_CUDA_TESTS");
    }
    int use_cuaev = std::get<1>(device_spec);
    std::string device_suffix = std::get<2>(device_spec);
    INFO("Testing model on device: " << "device" << device_suffix);
    INFO("cuAEV: " << (use_cuaev == 1));

    // Polarizabilities of H and C in the config are in AU. A non-default
    // inv_pol_dielectric requires mlmm_coupling = 'simple-polarizable', which requires ML
    // atomic charges
    std::ostringstream mlmm_options;
    mlmm_options << std::setprecision(17)
                 << "  pol_H = "
                 << COUPLING_ATOMIC_ALPHAS[0] << ",\n"
                 << "  pol_C = "
                 << COUPLING_ATOMIC_ALPHAS[1] << ",\n"
                 << "  inv_pol_dielectric = " << COUPLING_INV_POL_DIELECTRIC << ",\n"
                 << "  mlmm_coupling = 'simple-polarizable', use_mlip_atomic_charges = T,\n";
    mlip_initialize(
        size,
        atomic_numbers,
        writeMlipConfig(
            model_type, network_index, use_double, use_cuda_device, use_cuaev, 0,
            mlmm_options.str()
        ).c_str(),
        true,
        "none",
        false
    );

    std::string values_path = file_directory + "test_values" + device_suffix
        + model_suffix + ".txt";
    std::ifstream values_file(values_path);
    REQUIRE(values_file.good());
    std::vector<double> test_values;
    double value;
    while (values_file >> value) {
        test_values.push_back(value);
    }
    bool model_is_ensemble = network_index == -1;
    bool model_has_charges = model_type == "animbis";
    int charges_offset = model_is_ensemble ? 14 : 7;

    SECTION("Energy + Force, external neighbors") {
        int neighborlist[2][1] = {{0}, {1}};
        double shifts[1][3] = {{0.0, 0.0, 0.0}};
        mlip_calc_energy_force_from_external_neighbors(
            size,
            1,
            coords,
            &neighborlist[0][0],
            shifts,
            molecule_idxs_buf,
            false,
            forces,
            &pot_energy
        );
        FLOAT_EQ(pot_energy, test_values[0], use_double);
        for (int j = 0; j != size; ++j) {
            for (int c = 0; c != 3; ++c) {
                FLOAT_EQ(forces[j][c], test_values[1 + 3 * j + c], use_double);
            }
        }
    }

    SECTION("Energy + Force + QBC") {
        if (model_is_ensemble) {
            mlip_initialize(
                size,
                atomic_numbers,
                writeMlipConfig(
                    model_type, network_index, use_double, use_cuda_device, use_cuaev
                ).c_str(),
                true,
                "none",
                false
            );
            double qbc;
            double qbc_deriv[size][3];
            double fixed_atomic_charges[size] = {0.0, 0.0};
            int charges_dims[1] = {size};
            mlip_store_float_tensor("fixed_atomic_charges", fixed_atomic_charges, charges_dims, 1);
            mlip_calc_energy_force_mlmm_qbc(
                size,
                /* mm_atoms_num= */ 0,
                coords,
                /* mm_coords= */ nullptr,
                /* mm_atomic_charges= */ nullptr,
                /* cell= */ nullptr,
                /* use_pbc= */ false,
                /* verbose= */ false,
                forces,
                /* mm_forces= */ nullptr,
                &pot_energy
            );
            int qbc_grad_dims[2] = {size, 3};
            mlip_fetch_result("qbc", &qbc, nullptr, 0);
            mlip_fetch_result("qbc_grad", &qbc_deriv[0][0], qbc_grad_dims, 2);
            FLOAT_EQ(pot_energy, test_values[0], use_double);
            for (int j = 0; j != size; ++j) {
                for (int c = 0; c != 3; ++c) {
                    FLOAT_EQ(forces[j][c], test_values[1 + 3 * j + c], use_double);
                }
            }
            // QBC is the std of large energies, float32 loses digits to cancellation
            if (use_double) {
                FLOAT_EQ(qbc, test_values[7], true);
                for (int j = 0; j != size; ++j) {
                    for (int c = 0; c != 3; ++c) {
                        FLOAT_EQ(qbc_deriv[j][c], test_values[8 + 3 * j + c], true);
                    }
                }
            }
        }
    }

    SECTION("Energy + Force + Atomic charges") {
        if (model_has_charges) {
            double atomic_charges[size];
            double cell[3][3] = {{10., 0., 0.}, {0., 10., 0.}, {0., 0., 10.}};
            mlip_calc_energy_force(
                size, coords, cell, false, molecule_idxs_buf, false, forces, &pot_energy
            );
            int charges_dims[1] = {size};
            mlip_fetch_result("atomic_charges", atomic_charges, charges_dims, 1);
            FLOAT_EQ(pot_energy, test_values[0], use_double);
            for (int j = 0; j != size; ++j) {
                FLOAT_EQ(atomic_charges[j], test_values[charges_offset + j], use_double);
                for (int c = 0; c != 3; ++c) {
                    FLOAT_EQ(forces[j][c], test_values[1 + 3 * j + c], use_double);
                }
            }
        }
    }

    SECTION("Energy + Force with coupling") {
        if (model_has_charges) {
            double mm_coords[2][3] = {{5.0, 4.0, 2.0}, {0.0, 0.0, 3.0}};
            CouplingOutput out = calc_coupling(coords, mm_coords);
            for (int j = 0; j != size; ++j) {
                FLOAT_EQ(
                    out.ml_atomic_charges[j], test_values[charges_offset + j], use_double
                );
            }

            double expect_coulomb = 0.0;
            double expect_pol = 0.0;
            for (int j = 0; j != size; ++j) {
                double efield[3] = {0.0, 0.0, 0.0};
                for (int k = 0; k != 2; ++k) {
                    double delta[3];
                    double dist_sq = 0.0;
                    for (int c = 0; c != 3; ++c) {
                        delta[c] = mm_coords[k][c] - coords[j][c];
                        dist_sq += delta[c] * delta[c];
                    }
                    double dist = std::sqrt(dist_sq);
                    expect_coulomb += COUPLING_MM_ATOMIC_CHARGES[k] * out.ml_atomic_charges[j]
                        / (dist * constants::CODATA08_ANGSTROM_TO_BOHR);
                    for (int c = 0; c != 3; ++c) {
                        efield[c] += COUPLING_MM_ATOMIC_CHARGES[k] * delta[c]
                            / (dist_sq * dist
                               * std::pow(constants::CODATA08_ANGSTROM_TO_BOHR, 2));
                    }
                }
                double efield_sq =
                    efield[0] * efield[0] + efield[1] * efield[1] + efield[2] * efield[2];
                expect_pol -= COUPLING_INV_POL_DIELECTRIC * COUPLING_ATOMIC_ALPHAS[j]
                    * efield_sq;
            }
            // The in vacuo energy is the reference energy of the model
            double expect_embed = expect_coulomb + expect_pol - 0.5 * expect_pol;
            FLOAT_EQ(
                out.pot_energy,
                test_values[0] + expect_embed * constants::HARTREE_TO_KCALMOL,
                use_double
            );

            std::stringstream captured;
            std::streambuf* cout_buf = std::cout.rdbuf(captured.rdbuf());
            calc_coupling(coords, mm_coords, true);
            std::cout.rdbuf(cout_buf);
            CHECK(captured.str().find("ML/MM: IN VACUO ENERGY") != std::string::npos);
            CHECK(
                captured.str().find("ML/MM: ML-MM COUPLING ENERGY (TOTAL)")
                != std::string::npos
            );

            // With charge derivatives, forces are minus the gradient of the total
            // energy (checked by central finite differences)
            if (use_double) {
                const double step = 1e-4;
                for (int j = 0; j != 2; ++j) {
                    for (int c = 0; c != 3; ++c) {
                        double displaced[2][3];
                        std::memcpy(displaced, coords, sizeof(displaced));
                        displaced[j][c] += step;
                        double ene_plus =
                            calc_coupling(displaced, mm_coords).pot_energy;
                        displaced[j][c] -= 2 * step;
                        double ene_minus =
                            calc_coupling(displaced, mm_coords).pot_energy;
                        CHECK_THAT(
                            out.ml_forces[j][c],
                            WithinAbs(-(ene_plus - ene_minus) / (2 * step), 1e-5)
                        );

                        std::memcpy(displaced, mm_coords, sizeof(displaced));
                        displaced[j][c] += step;
                        ene_plus = calc_coupling(coords, displaced).pot_energy;
                        displaced[j][c] -= 2 * step;
                        ene_minus = calc_coupling(coords, displaced).pot_energy;
                        CHECK_THAT(
                            out.mm_forces[j][c],
                            WithinAbs(-(ene_plus - ene_minus) / (2 * step), 1e-5)
                        );
                    }
                }
            }
        }
    }

    SECTION("Energy + Force with no coupling") {
        double mm_coords[2][3] = {{5.0, 4.0, 2.0}, {0.0, 0.0, 3.0}};
        double mm_atomic_charges[2] = {
            COUPLING_MM_ATOMIC_CHARGES[0], COUPLING_MM_ATOMIC_CHARGES[1]
        };
        double ml_forces[size][3];
        double mm_forces[2][3];
        double pot_energy;
        // Case-insensitive
        mlip_initialize(
            size,
            atomic_numbers,
            writeMlipConfig(
                model_type, network_index, use_double, use_cuda_device, use_cuaev, 0,
                "  mlmm_coupling = 'None',\n"
            ).c_str(),
            true,
            "none",
            false
        );
        mlip_calc_energy_force_mlmm(
            size,
            2,
            coords,
            mm_coords,
            mm_atomic_charges,
            /* cell= */ nullptr,
            /* use_pbc= */ false,
            /* verbose= */ false,
            ml_forces,
            mm_forces,
            &pot_energy
        );
        FLOAT_EQ(pot_energy, test_values[0], use_double);
        for (int j = 0; j != size; ++j) {
            for (int c = 0; c != 3; ++c) {
                FLOAT_EQ(ml_forces[j][c], test_values[1 + 3 * j + c], use_double);
                CHECK(mm_forces[j][c] == 0.0);
            }
        }
    }

    SECTION("Energy + Force with custom ML/MM model") {
        if (model_has_charges) {
            double mm_coords[2][3] = {{5.0, 4.0, 2.0}, {0.0, 0.0, 3.0}};
            double mm_atomic_charges[2] = {
                COUPLING_MM_ATOMIC_CHARGES[0], COUPLING_MM_ATOMIC_CHARGES[1]
            };
            double cell[3][3] = {{10., 0., 0.}, {0., 10., 0.}, {0., 0., 10.}};
            // Reference: animbis, mlmm_coupling='simple-polarizable',
            // use_torchani_charges=.true., with the
            // default (Amber) polarizabilities
            mlip_initialize(
                size,
                atomic_numbers,
                writeMlipConfig(
                    model_type, network_index, use_double, use_cuda_device, use_cuaev, 0,
                    "  mlmm_coupling = 'simple-polarizable', use_mlip_atomic_charges = T,\n"
                ).c_str(),
                true,
                "none",
                false
            );
            CouplingOutput expect{};
            mlip_calc_energy_force_mlmm(
                size,
                2,
                coords,
                mm_coords,
                mm_atomic_charges,
                /* cell= */ nullptr,
                /* use_pbc= */ false,
                /* verbose= */ false,
                expect.ml_forces,
                expect.mm_forces,
                &expect.pot_energy
            );

            mlip_initialize(
                size,
                atomic_numbers,
                writeMlipConfig(
                    "animbis-custom-mlmm", network_index, use_double, use_cuda_device, use_cuaev,
                    0, "  mlmm_coupling = 'custom',\n"
                ).c_str(),
                true,
                "none",
                false
            );
            double ml_forces[size][3];
            double mm_forces[2][3];
            double pot_energy;
            mlip_calc_energy_force_mlmm(
                size,
                2,
                coords,
                mm_coords,
                mm_atomic_charges,
                cell,
                /* use_pbc= */ false,
                /* verbose= */ false,
                ml_forces,
                mm_forces,
                &pot_energy
            );
            FLOAT_EQ(pot_energy, expect.pot_energy, use_double);
            for (int j = 0; j != size; ++j) {
                for (int c = 0; c != 3; ++c) {
                    FLOAT_EQ(ml_forces[j][c], expect.ml_forces[j][c], use_double);
                    FLOAT_EQ(
                        mm_forces[j][c],
                        expect.mm_forces[j][c],
                        use_double
                    );
                }
            }
        }
    }
}


TEST_CASE("Charges for md engine electrostatics") {
    // AIMNet2 on the CPU, compared with the other entry points of the same model.
    // AIMNet2 keeps part of its arithmetic in float32 even in double precision,
    // so different backward passes through the same graph only agree to float32
    // round-off: float32 tolerances are used for both precisions.
    // The AIMNet2 models are downloaded at configure time (TORCHANI_DOWNLOAD_JIT_MODELS).
    std::string model_path = std::string(TORCHANI_MLIPS_BUILD_DIR)
        + "/aimnet2/aimnet2-wb97m-nocut.pt";
    if (not std::ifstream(model_path).good()) {
        SKIP(
            "AIMNet2 model not found (" << model_path << "): configure with"
            << " -DTORCHANI_DOWNLOAD_JIT_MODELS=ON"
        );
    }
    int use_double = GENERATE(0, 1);
    INFO("Double precision: " << use_double);
    double dummy_cell[3][3] = {{0., 0., 0.}, {0., 0., 0.}, {0., 0., 0.}};

    SECTION("Neutral water: retain + vjp vs full charge derivatives") {
        const int size = 3;
        double coords[size][3] = {{0.0, 0.0, 0.1}, {0.0, 0.76, -0.47}, {0.05, -0.74, -0.5}};
        int atomic_numbers[size] = {8, 1, 1};
        int no_eval_atoms[size] = {0, 0, 0};
        double ml_electrostatic_potential[size] = {12.5, -30.0, 7.25};  // kcal/mol/e
        double zero_ml_electrostatic_potential[size] = {0.0, 0.0, 0.0};
        mlip_initialize(
            size,
            atomic_numbers,
            writeMlipConfig("aimnet2-wb97m-nocut", -1, use_double, false, false).c_str(),
            false,
            "none",
            false
        );
        double charges[size], energy;
        double ref_forces[size][3], ref_charges[size], ref_energy;
        int molecule_idxs[size] = {0, 0, 0};
        mlip_calc_energy_force(
            size, coords, dummy_cell, false, molecule_idxs, false, ref_forces, &ref_energy
        );
        int charges_dims[1] = {size};
        mlip_fetch_result("atomic_charges", ref_charges, charges_dims, 1);
        std::vector<double> dq =
            calcAtomicChargeGrads(model_path, size, atomic_numbers, coords, 0, use_double);
        // Zero potential: the in-vacuo forces
        double vjp_forces[size][3];
        mlip_calc_energy_atomic_charges_retain(size, coords, charges, &energy);
        FLOAT_EQ(energy, ref_energy, false);
        for (int j = 0; j != size; ++j) {
            FLOAT_EQ(charges[j], ref_charges[j], false);
        }
        mlip_calc_energy_atomic_charges_vjp(size, zero_ml_electrostatic_potential, vjp_forces);
        for (int j = 0; j != size; ++j) {
            for (int c = 0; c != 3; ++c) {
                FLOAT_EQ(vjp_forces[j][c], ref_forces[j][c], false);
            }
        }
        // The in-vacuo forces plus those of the charges:
        // dq[c + 3 * j + 3 * size * k] = d(charge k) / d(coordinate c of atom j)
        mlip_calc_energy_atomic_charges_retain(size, coords, charges, &energy);
        mlip_calc_energy_atomic_charges_vjp(size, ml_electrostatic_potential, vjp_forces);
        for (int j = 0; j != size; ++j) {
            for (int c = 0; c != 3; ++c) {
                double expect = ref_forces[j][c];
                for (int k = 0; k != size; ++k) {
                    expect -= ml_electrostatic_potential[k] * dq[c + 3 * j + 3 * size * k];
                }
                // Round-off error is amplified by the potential applied in the vj product
                // So increase tolerance a little above float precision
                CHECK_THAT(
                    vjp_forces[j][c],
                    WithinAbs(expect, 1e-4) || WithinRel(expect, 1e-4)
                );
            }
        }
    }

    SECTION("Polarizabilities in use are printed") {
        const int size = 3;
        int atomic_numbers[size] = {8, 1, 1};
        std::stringstream captured;
        std::streambuf* cout_buf = std::cout.rdbuf(captured.rdbuf());
        mlip_initialize(
            size,
            atomic_numbers,
            writeMlipConfig(
                "aimnet2-wb97m-nocut", -1, use_double, false, false, 0,
                "  mlmm_coupling = 'simple-polarizable', use_mlip_atomic_charges = T,\n"
                "  pol_O = 6.0,\n"
            ).c_str(),
            true,
            "none",
            true
        );
        std::cout.rdbuf(cout_buf);
        std::string info = captured.str();
        CHECK(info.find("atomic polarizabilities (alpha) are in use") != std::string::npos);
        CHECK(info.find(" H, 3.0800") != std::string::npos);
        CHECK(info.find(" O, 6.0000") != std::string::npos);
    }

    SECTION("Ammonium: net charge") {
        const int size = 5;
        double coords[size][3] = {
            {0.0, 0.0, 0.0}, {0.6, 0.6, 0.6}, {-0.6, -0.6, 0.6},
            {-0.6, 0.6, -0.6}, {0.6, -0.6, -0.65}
        };
        int atomic_numbers[size] = {7, 1, 1, 1, 1};
        int no_eval_atoms[size] = {0, 0, 0, 0, 0};
        int molecule_idxs[size] = {0, 0, 0, 0, 0};
        mlip_initialize(
            size,
            atomic_numbers,
            writeMlipConfig(
                "aimnet2-wb97m-nocut", -1, use_double, false, false, 0, "  ml_charge = 1,\n"
            ).c_str(),
            false,
            "none",
            false
        );
        double charges[size], energy;
        double ref_forces[size][3], ref_energy;
        mlip_calc_energy_force(
            size, coords, dummy_cell, false, molecule_idxs, false, ref_forces, &ref_energy
        );
        mlip_calc_energy_atomic_charges_retain(size, coords, charges, &energy);
        FLOAT_EQ(energy, ref_energy, false);
        double total = 0.0;
        for (int j = 0; j != size; ++j) {
            total += charges[j];
        }
        FLOAT_EQ(total, 1.0, false);
        // Uniform potential: the total charge is fixed, so only the in-vacuo forces
        double ml_electrostatic_potential[size] = {5.0, 5.0, 5.0, 5.0, 5.0};
        double vjp_forces[size][3];
        mlip_calc_energy_atomic_charges_vjp(size, ml_electrostatic_potential, vjp_forces);
        for (int j = 0; j != size; ++j) {
            for (int c = 0; c != 3; ++c) {
                FLOAT_EQ(vjp_forces[j][c], ref_forces[j][c], false);
            }
        }
    }
}

TEST_CASE("Tensor container", "[tensors]") {
    tensor_utils::TensorContainer container;
    CHECK_FALSE(container.contains("x"));
    CHECK_THROWS_AS(container.get("x"), std::out_of_range);
    container.store("x", torch::ones({2, 3}));
    REQUIRE(container.contains("x"));
    CHECK(container.get("x").sizes() == torch::IntArrayRef({2, 3}));
    container.store("x", torch::zeros({4}));
    CHECK(container.get("x").sizes() == torch::IntArrayRef({4}));
    CHECK(container.get("x").sum().item<double>() == 0.0);
    container.store("y", torch::ones({1}));
    container.clear("x");
    CHECK_FALSE(container.contains("x"));
    CHECK(container.contains("y"));
    CHECK_NOTHROW(container.clear("x"));
    container.clear();
    CHECK_FALSE(container.contains("y"));
}

TEST_CASE("Store tensors through the C API", "[tensors]") {
    int atomic_numbers[2] = {1, 6};
    mlip_initialize(
        2,
        atomic_numbers,
        writeMlipConfig("ani1x", 0, true, false, false).c_str(),
        false,
        "none",
        false
    );
    double float_values[2][3] = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    int int_values[3] = {7, 8, 9};
    int float_dims[2] = {2, 3};
    int int_dims[1] = {3};
    CHECK_NOTHROW(mlip_store_float_tensor("f", &float_values[0][0], float_dims, 2));
    CHECK_NOTHROW(mlip_store_int_tensor("i", int_values, int_dims, 1));
    // Storing again with the same name replaces the tensor
    CHECK_NOTHROW(mlip_store_int_tensor("i", int_values, int_dims, 1));
    int mask_values[3] = {1, 0, 1};
    CHECK_NOTHROW(mlip_store_mask_tensor("m", mask_values, int_dims, 1));
}

TEST_CASE("No-eval atoms mask", "[no_eval]") {
    // A water molecule, and a methane molecule far from it
    const int all_size = 8;
    double all_coords[all_size][3] = {
        {0.0, 0.0, 0.1}, {0.0, 0.76, -0.47}, {0.05, -0.74, -0.5},
        {20.0, 0.0, 0.0}, {20.63, 0.63, 0.63}, {19.37, -0.63, 0.63},
        {19.37, 0.63, -0.63}, {20.63, -0.63, -0.63},
    };
    int all_atomic_numbers[all_size] = {8, 1, 1, 6, 1, 1, 1, 1};
    int all_molecule_idxs[all_size] = {0, 0, 0, 1, 1, 1, 1, 1};
    int no_eval_atoms_mask[all_size] = {0, 0, 0, 1, 1, 1, 1, 1};
    double cell[3][3] = {{50., 0., 0.}, {0., 50., 0.}, {0., 0., 50.}};
    std::string config_path = writeMlipConfig("ani1x", 0, true, false, false);

    const int water_size = 3;
    double water_forces[water_size][3];
    double water_energy;
    mlip_initialize(
        water_size, all_atomic_numbers, config_path.c_str(), false, "none", false
    );
    mlip_calc_energy_force(
        water_size, all_coords, cell, false, all_molecule_idxs, false,
        water_forces, &water_energy
    );

    double forces[all_size][3];
    double energy;
    mlip_initialize(all_size, all_atomic_numbers, config_path.c_str(), false, "none", false);
    mlip_set_no_eval_atoms_mask(all_size, no_eval_atoms_mask);
    mlip_calc_energy_force(
        all_size, all_coords, cell, false, all_molecule_idxs, false, forces, &energy
    );
    CHECK_THAT(energy, Catch::Matchers::WithinAbs(water_energy, 1e-10));
    for (int atom = 0; atom != all_size; ++atom) {
        for (int c = 0; c != 3; ++c) {
            double expect_force = 0.0;
            if (atom < water_size) {
                expect_force = water_forces[atom][c];
            }
            CHECK_THAT(forces[atom][c], Catch::Matchers::WithinAbs(expect_force, 1e-10));
        }
    }

    // An all-zeros mask evaluates all atoms again
    int all_zeros_mask[all_size] = {0, 0, 0, 0, 0, 0, 0, 0};
    mlip_set_no_eval_atoms_mask(all_size, all_zeros_mask);
    mlip_calc_energy_force(
        all_size, all_coords, cell, false, all_molecule_idxs, false, forces, &energy
    );
    CHECK(std::abs(energy - water_energy) > 1.0);
}

TEST_CASE("Set ML charge and spin multiplicity", "[config]") {
    int atomic_numbers[2] = {1, 6};
    mlip_initialize(
        2,
        atomic_numbers,
        writeMlipConfig("ani1x", 0, true, false, false).c_str(),
        false,
        "none",
        false
    );
    int charge = 0;
    int spin_mult = 0;
    mlip_get_config_int_option("ml_charge", &charge);
    mlip_get_config_int_option("ml_spin_mult", &spin_mult);
    CHECK(charge == 0);
    CHECK(spin_mult == 1);
    mlip_set_ml_charge_and_spin_mult(-1, 2);
    mlip_get_config_int_option("ml_charge", &charge);
    mlip_get_config_int_option("ml_spin_mult", &spin_mult);
    CHECK(charge == -1);
    CHECK(spin_mult == 2);
}

TEST_CASE("Models without spin_mult only support spin_mult = 1", "[spin_mult]") {
    // Initializing loads the compiled extensions that the model needs
    int atomic_numbers[2] = {1, 6};
    mlip_initialize(
        2,
        atomic_numbers,
        writeMlipConfig("ani1x", 0, true, false, false).c_str(),
        false,
        "none",
        false
    );
    std::unique_ptr<mlip::MlipModel> model = mlip::MlipModel::from_model_path(
        std::string(TORCHANI_MLIPS_BUILD_DIR) + "/ani1x.pt", torch::kCPU, torch::kDouble
    );
    torch::Tensor species = torch::tensor({{1, 6}}, torch::kLong);
    torch::Tensor coords =
        torch::tensor({{{3.0, 3.0, 4.0}, {1.0, 2.0, 1.0}}}, torch::kDouble);
    CHECK_NOTHROW(model->computeSinglePoint(species, coords, std::nullopt, 0, 1));
    CHECK_THROWS_WITH(
        model->computeSinglePoint(species, coords, std::nullopt, 0, 3),
        Catch::Matchers::ContainsSubstring("only spin_mult = 1 is supported")
    );
}

TEST_CASE("Output files", "[io]") {
    const int size = 2;
    int atomic_numbers[size] = {1, 6};
    double coords[size][3] = {{3.0, 3.0, 4.0}, {1.0, 2.0, 1.0}};
    double cell[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    int molecule_idxs[size] = {0, 0};
    double forces[size][3];
    double energy;
    std::filesystem::path output_dir =
        std::filesystem::temp_directory_path() / "torchani_test_outputs";
    std::filesystem::remove_all(output_dir);
    mlip_initialize(
        size,
        atomic_numbers,
        writeMlipConfig(
            "animbis", 0, true, false, false, 0,
            "  write_coords = T, write_forces = T, write_atomic_charges = T,\n"
            "  write_interval = 2, output_dir = '" + output_dir.string() + "',\n"
        ).c_str(),
        false,
        "none",
        false
    );
    // Outputs are written at steps 2 and 4
    for (int step = 1; step != 6; ++step) {
        mlip_update_md_step(step);
        mlip_calc_energy_force(size, coords, cell, false, molecule_idxs, false, forces, &energy);
    }
    // Outputs are written only once per step
    mlip_update_md_step(4);
    mlip_calc_energy_force(size, coords, cell, false, molecule_idxs, false, forces, &energy);
    auto read_lines = [](const std::filesystem::path& path) {
        std::ifstream file(path);
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(file, line)) {
            lines.push_back(line);
        }
        return lines;
    };
    // Two frames of: atom count, comment, one line per atom
    std::vector<std::string> xyz = read_lines(output_dir / "coords.xyz");
    REQUIRE(xyz.size() == 8);
    CHECK(xyz[0] == "       2");
    CHECK(xyz[2] == "  1      3.0000000      3.0000000      4.0000000");
    // Two frames of: one line per atom, and an empty line
    std::vector<std::string> forces_lines = read_lines(output_dir / "forces.mlip.dat");
    REQUIRE(forces_lines.size() == 6);
    CHECK(forces_lines[2].empty());
    double fx = std::stod(forces_lines[0].substr(0, 15));
    CHECK_THAT(fx, Catch::Matchers::WithinAbs(forces[0][0], 1e-6));
    CHECK(read_lines(output_dir / "atomic_charges.mlip.dat").size() == 6);
    CHECK_FALSE(std::filesystem::exists(output_dir / "ml_region.xyz"));
    std::filesystem::remove_all(output_dir);
}

TEST_CASE("ML/MM output files", "[io]") {
    const int size = 2;
    int atomic_numbers[size] = {1, 6};
    double coords[size][3] = {{3.0, 3.0, 4.0}, {1.0, 2.0, 1.0}};
    double mm_coords[1][3] = {{6.0, 6.0, 6.0}};
    double mm_atomic_charges[1] = {0.5};
    double ml_forces[size][3];
    double mm_forces[1][3];
    double energy;
    std::filesystem::path output_dir =
        std::filesystem::temp_directory_path() / "torchani_test_mlmm_outputs";
    std::filesystem::remove_all(output_dir);
    mlip_initialize(
        size,
        atomic_numbers,
        writeMlipConfig(
            "animbis", 0, true, false, false, 0,
            "  mlmm_coupling = 'simple-polarizable', use_mlip_atomic_charges = T,\n"
            "  write_coords = T, write_forces = T, write_atomic_charges = T,\n"
            "  output_dir = '" + output_dir.string() + "',\n"
        ).c_str(),
        true,
        "none",
        false
    );
    for (int step = 1; step != 3; ++step) {
        mlip_update_md_step(step);
        mlip_calc_energy_force_mlmm(
            size, 1, coords, mm_coords, mm_atomic_charges, nullptr, false, false,
            ml_forces, mm_forces, &energy
        );
    }
    auto read_lines = [](const std::filesystem::path& path) {
        std::ifstream file(path);
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(file, line)) {
            lines.push_back(line);
        }
        return lines;
    };
    // Two frames of: atom count, comment, one line per atom
    std::vector<std::string> xyz = read_lines(output_dir / "ml_region.xyz");
    REQUIRE(xyz.size() == 8);
    CHECK(xyz[0] == "       2");
    CHECK(xyz[2] == "  1      3.0000000      3.0000000      4.0000000");
    // Two frames of: one line per atom, and an empty line
    std::vector<std::string> ml_forces_lines =
        read_lines(output_dir / "forces_ml_region.mlip.dat");
    REQUIRE(ml_forces_lines.size() == 6);
    CHECK(ml_forces_lines[2].empty());
    double fx = std::stod(ml_forces_lines[0].substr(0, 15));
    CHECK_THAT(fx, Catch::Matchers::WithinAbs(ml_forces[0][0], 1e-6));
    CHECK(read_lines(output_dir / "forces_mlmm_mm_region.mlip.dat").size() == 4);
    CHECK(read_lines(output_dir / "atomic_charges_ml_region.mlip.dat").size() == 6);
    std::filesystem::remove_all(output_dir);
}
