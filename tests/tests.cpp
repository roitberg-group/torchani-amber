#include <torchani.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

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
        int use_cuaev = std::get<2>(device_spec); // Only enabled for cuda devices
        std::string device_suffix = std::get<3>(device_spec);
        INFO("Testing model on device: " << "device" << device_suffix);
        INFO("cuAEV: " << (use_cuaev == 1));
        // TODO: Cell seems to be too small for pbc, but this should generate
        // tests for the cell too

        // output
        double potential_energy;

        torchani_initialize(
            size,
            atomic_numbers,
            no_eval_atoms,
            device_index,
            model_type.c_str(),
            network_index,
            use_double,
            use_cuda_device,
            use_cuaev
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
                torchani_calc_energy_force(
                    size,
                    coords,
                    cell,
                    true,
                    molecule_idxs_buf,
                    false,
                    0,
                    forces,
                    &potential_energy
                );
                FLOAT_EQ(potential_energy, test_values[0], use_double);
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
            double* atomic_charge_derivatives;
            atomic_charge_derivatives = (double*) malloc(2 * 2 * 3 * sizeof(double));
            for (long j = 0; j != 10; ++j){
                torchani_calc_data_for_monitored_mlmm(
                    size,
                    coords,
                    0,
                    /* outputs */
                    forces,
                    atomic_charges,
                    atomic_charge_derivatives,
                    &qbc,
                    qbc_deriv,
                    &potential_energy
                );
                FLOAT_EQ(potential_energy, test_values[0], use_double);
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
            free(atomic_charge_derivatives);
        }
    }
}

static const double ANGSTROM_TO_BOHR = 1.8897261328727875;
static const double HARTREE_TO_KCALMOL = 627.5094738898777;
static const double COUPLING_INV_POL_DIELECTRIC = 0.8;
static const double COUPLING_ATOMIC_ALPHAS[] = {0.5, 1.2};
static const double COUPLING_ENV_CHARGES[] = {0.5, -0.4};

struct CouplingOutput {
    double forces_on_atoms[2][3];
    double forces_on_env_charges[2][3];
    double atomic_charges[2];
    double ene_pot_invacuo;
    double ene_pot_embed_pol;
    double ene_pot_embed_dist;
    double ene_pot_embed_coulomb;
    double ene_pot_total;
};

// Amber defaults for mlmm_coupling=1 (H and C polarizabilities in AU, converted to
// Ang^3 as in the Amber patch)
static const double AMBER_INV_POL_DIELECTRIC = 0.5;
static const double AMBER_AU_TO_ANG3 = 0.1481847;
static const double AMBER_ATOMIC_ALPHAS_AU[] = {3.08, 11.3};

static CouplingOutput calc_coupling(
    double coords[][3], double env_charge_coords[][3], bool use_charge_derivatives
) {
    double atomic_alphas[2] = {COUPLING_ATOMIC_ALPHAS[0], COUPLING_ATOMIC_ALPHAS[1]};
    double env_charges[2] = {COUPLING_ENV_CHARGES[0], COUPLING_ENV_CHARGES[1]};
    CouplingOutput out{};
    torchani_calc_energy_force_with_coupling(
        2,
        2,
        COUPLING_INV_POL_DIELECTRIC,
        coords,
        atomic_alphas,
        env_charge_coords,
        env_charges,
        /* predict_charges= */ true,
        /* use_simple_polarization_correction= */ true,
        use_charge_derivatives,
        /* ml_system_charge= */ 0,
        out.forces_on_atoms,
        out.forces_on_env_charges,
        out.atomic_charges,
        &out.ene_pot_invacuo,
        &out.ene_pot_embed_pol,
        &out.ene_pot_embed_dist,
        &out.ene_pot_embed_coulomb,
        &out.ene_pot_total
    );
    return out;
}

TEST_CASE("C bindings for other entry points") {
    const int size = 2;
    double coords[size][3] = {{3.0, 3.0, 4.0}, {1.0, 2.0, 1.0}};
    int molecule_idxs_buf[size] = {1, 1};
    int atomic_numbers[size] = {1, 6};
    int no_eval_atoms[size] = {0, 0};
    double forces[size][3];
    double potential_energy;
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

    torchani_initialize(
        size,
        atomic_numbers,
        no_eval_atoms,
        0,
        model_type.c_str(),
        network_index,
        use_double,
        use_cuda_device,
        use_cuaev
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
        torchani_calc_energy_force_from_external_neighbors(
            size,
            1,
            coords,
            &neighborlist[0][0],
            shifts,
            molecule_idxs_buf,
            false,
            0,
            forces,
            &potential_energy
        );
        FLOAT_EQ(potential_energy, test_values[0], use_double);
        for (int j = 0; j != size; ++j) {
            for (int c = 0; c != 3; ++c) {
                FLOAT_EQ(forces[j][c], test_values[1 + 3 * j + c], use_double);
            }
        }
    }

    SECTION("Energy + Force + QBC") {
        if (model_is_ensemble) {
            double qbc;
            double qbc_deriv[size][3];
            torchani_calc_energy_force_qbc(
                size, coords, 0, forces, &qbc, qbc_deriv, &potential_energy
            );
            FLOAT_EQ(potential_energy, test_values[0], use_double);
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
            torchani_energy_force_atomic_charges(
                size, coords, forces, atomic_charges, &potential_energy
            );
            FLOAT_EQ(potential_energy, test_values[0], use_double);
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
            double env_charge_coords[2][3] = {{5.0, 4.0, 2.0}, {0.0, 0.0, 3.0}};
            CouplingOutput out = calc_coupling(coords, env_charge_coords, true);
            FLOAT_EQ(out.ene_pot_invacuo, test_values[0], use_double);
            for (int j = 0; j != size; ++j) {
                FLOAT_EQ(
                    out.atomic_charges[j], test_values[charges_offset + j], use_double
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
                        delta[c] = env_charge_coords[k][c] - coords[j][c];
                        dist_sq += delta[c] * delta[c];
                    }
                    double dist = std::sqrt(dist_sq);
                    expect_coulomb += COUPLING_ENV_CHARGES[k] * out.atomic_charges[j]
                        / (dist * ANGSTROM_TO_BOHR);
                    for (int c = 0; c != 3; ++c) {
                        efield[c] += COUPLING_ENV_CHARGES[k] * delta[c]
                            / (dist_sq * dist * ANGSTROM_TO_BOHR * ANGSTROM_TO_BOHR);
                    }
                }
                double efield_sq =
                    efield[0] * efield[0] + efield[1] * efield[1] + efield[2] * efield[2];
                expect_pol -= COUPLING_INV_POL_DIELECTRIC * COUPLING_ATOMIC_ALPHAS[j]
                    * std::pow(ANGSTROM_TO_BOHR, 3) * efield_sq;
            }
            FLOAT_EQ(
                out.ene_pot_embed_coulomb,
                expect_coulomb * HARTREE_TO_KCALMOL,
                use_double
            );
            FLOAT_EQ(out.ene_pot_embed_pol, expect_pol * HARTREE_TO_KCALMOL, use_double);
            FLOAT_EQ(out.ene_pot_embed_dist, -0.5 * out.ene_pot_embed_pol, use_double);
            FLOAT_EQ(
                out.ene_pot_total,
                out.ene_pot_invacuo + out.ene_pot_embed_pol + out.ene_pot_embed_dist
                    + out.ene_pot_embed_coulomb,
                use_double
            );

            // Charges depend only on the ML coords, so forces on the env charges
            // don't depend on use_charge_derivatives
            CouplingOutput out_no_deriv = calc_coupling(coords, env_charge_coords, false);
            FLOAT_EQ(out_no_deriv.ene_pot_total, out.ene_pot_total, use_double);
            for (int k = 0; k != 2; ++k) {
                for (int c = 0; c != 3; ++c) {
                    FLOAT_EQ(
                        out_no_deriv.forces_on_env_charges[k][c],
                        out.forces_on_env_charges[k][c],
                        use_double
                    );
                }
            }

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
                            calc_coupling(displaced, env_charge_coords, true).ene_pot_total;
                        displaced[j][c] -= 2 * step;
                        double ene_minus =
                            calc_coupling(displaced, env_charge_coords, true).ene_pot_total;
                        CHECK_THAT(
                            out.forces_on_atoms[j][c],
                            WithinAbs(-(ene_plus - ene_minus) / (2 * step), 1e-5)
                        );

                        std::memcpy(displaced, env_charge_coords, sizeof(displaced));
                        displaced[j][c] += step;
                        ene_plus = calc_coupling(coords, displaced, true).ene_pot_total;
                        displaced[j][c] -= 2 * step;
                        ene_minus = calc_coupling(coords, displaced, true).ene_pot_total;
                        CHECK_THAT(
                            out.forces_on_env_charges[j][c],
                            WithinAbs(-(ene_plus - ene_minus) / (2 * step), 1e-5)
                        );
                    }
                }
            }
        }
    }

    SECTION("Energy + Force with custom ML/MM model") {
        if (model_has_charges) {
            double env_charge_coords[2][3] = {{5.0, 4.0, 2.0}, {0.0, 0.0, 3.0}};
            double env_charges[2] = {COUPLING_ENV_CHARGES[0], COUPLING_ENV_CHARGES[1]};
            double atomic_alphas[2] = {
                AMBER_ATOMIC_ALPHAS_AU[0] * AMBER_AU_TO_ANG3,
                AMBER_ATOMIC_ALPHAS_AU[1] * AMBER_AU_TO_ANG3,
            };
            double cell[3][3] = {{10., 0., 0.}, {0., 10., 0.}, {0., 0., 10.}};
            // Reference: animbis, mlmm_coupling=1, use_torchani_charges=.true.
            CouplingOutput expect{};
            torchani_calc_energy_force_with_coupling(
                size,
                2,
                AMBER_INV_POL_DIELECTRIC,
                coords,
                atomic_alphas,
                env_charge_coords,
                env_charges,
                /* predict_charges= */ true,
                /* use_simple_polarization_correction= */ true,
                /* use_charge_derivatives= */ true,
                /* ml_system_charge= */ 0,
                expect.forces_on_atoms,
                expect.forces_on_env_charges,
                expect.atomic_charges,
                &expect.ene_pot_invacuo,
                &expect.ene_pot_embed_pol,
                &expect.ene_pot_embed_dist,
                &expect.ene_pot_embed_coulomb,
                &expect.ene_pot_total
            );

            torchani_initialize(
                size,
                atomic_numbers,
                no_eval_atoms,
                0,
                "animbis-custom-mlmm",
                network_index,
                use_double,
                use_cuda_device,
                use_cuaev
            );
            bool calc_only_bonded = GENERATE(false, true);
            INFO("calc_only_bonded: " << calc_only_bonded);
            double forces_on_atoms[size][3];
            double forces_on_env_charges[2][3];
            double ene_pot_total;
            torchani_calc_energy_force_custom_mlmm(
                size,
                2,
                coords,
                env_charge_coords,
                env_charges,
                cell,
                /* use_pbc= */ false,
                molecule_idxs_buf,
                calc_only_bonded,
                /* ml_system_charge= */ 0,
                forces_on_atoms,
                forces_on_env_charges,
                &ene_pot_total
            );
            FLOAT_EQ(ene_pot_total, expect.ene_pot_total, use_double);
            for (int j = 0; j != size; ++j) {
                for (int c = 0; c != 3; ++c) {
                    FLOAT_EQ(forces_on_atoms[j][c], expect.forces_on_atoms[j][c], use_double);
                    FLOAT_EQ(
                        forces_on_env_charges[j][c],
                        expect.forces_on_env_charges[j][c],
                        use_double
                    );
                }
            }
        }
    }
}


TEST_CASE("Charges for host electrostatics") {
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
        double weights[size] = {12.5, -30.0, 7.25};  // kcal/mol/e
        double zero_weights[size] = {0.0, 0.0, 0.0};
        torchani_initialize(
            size, atomic_numbers, no_eval_atoms, 0, "aimnet2-wb97m-nocut", -1,
            use_double, false, false
        );
        double charges[size], energy;
        double ref_forces[size][3], ref_charges[size], ref_energy;
        double* dq = (double*) malloc(size * size * 3 * sizeof(double));
        torchani_energy_force_atomic_charges_with_derivatives(
            size, coords, ref_forces, ref_charges, dq, &ref_energy
        );
        // Zero weights: the in-vacuo forces
        double vjp_forces[size][3];
        torchani_energy_atomic_charges_retain(size, coords, 0, charges, &energy);
        FLOAT_EQ(energy, ref_energy, false);
        for (int j = 0; j != size; ++j) {
            FLOAT_EQ(charges[j], ref_charges[j], false);
        }
        torchani_energy_atomic_charges_vjp(size, zero_weights, vjp_forces);
        for (int j = 0; j != size; ++j) {
            for (int c = 0; c != 3; ++c) {
                FLOAT_EQ(vjp_forces[j][c], ref_forces[j][c], false);
            }
        }
        // The in-vacuo forces plus those of the charges:
        // dq[c + 3 * j + 3 * size * k] = d(charge k) / d(coordinate c of atom j)
        torchani_energy_atomic_charges_retain(size, coords, 0, charges, &energy);
        torchani_energy_atomic_charges_vjp(size, weights, vjp_forces);
        for (int j = 0; j != size; ++j) {
            for (int c = 0; c != 3; ++c) {
                double expect = ref_forces[j][c];
                for (int k = 0; k != size; ++k) {
                    expect -= weights[k] * dq[c + 3 * j + 3 * size * k];
                }
                // Round-off error is amplified by the weights applied in the vj product
                // So increase tolerance a little above float precision
                CHECK_THAT(
                    vjp_forces[j][c],
                    WithinAbs(expect, 1e-4) || WithinRel(expect, 1e-4)
                );
            }
        }
        free(dq);
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
        torchani_initialize(
            size, atomic_numbers, no_eval_atoms, 0, "aimnet2-wb97m-nocut", -1,
            use_double, false, false
        );
        double charges[size], energy;
        double ref_forces[size][3], ref_energy;
        torchani_calc_energy_force(
            size, coords, dummy_cell, false, molecule_idxs, false, 1,
            ref_forces, &ref_energy
        );
        torchani_energy_atomic_charges_retain(size, coords, 1, charges, &energy);
        FLOAT_EQ(energy, ref_energy, false);
        double total = 0.0;
        for (int j = 0; j != size; ++j) {
            total += charges[j];
        }
        FLOAT_EQ(total, 1.0, false);
        // Uniform weights: the total charge is fixed, so only the in-vacuo forces
        double weights[size] = {5.0, 5.0, 5.0, 5.0, 5.0};
        double vjp_forces[size][3];
        torchani_energy_atomic_charges_vjp(size, weights, vjp_forces);
        for (int j = 0; j != size; ++j) {
            for (int c = 0; c != 3; ++c) {
                FLOAT_EQ(vjp_forces[j][c], ref_forces[j][c], false);
            }
        }
    }
}
