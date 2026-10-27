#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include <tuple>
#include <fstream>
#include <iostream>
#include <string>
#include <cstdlib>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <torchani.h>
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
        // TODO: Ability to skip cuda tests
        int use_cuda_device = std::get<0>(device_spec);
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
            model_type.c_str(),
            device_index,
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

TEST_CASE("Charges for host electrostatics") {
    // AIMNet2 on the CPU, compared with the other entry points of the same model.
    // AIMNet2 keeps part of its arithmetic in float32 even in double precision,
    // so different backward passes through the same graph only agree to float32
    // round-off: float32 tolerances are used for both precisions.
    // The AIMNet2 models are generated by the build (JIT_COMPILE_AIMNET2) or by
    // "python jit/jit.py --aimnet2".
    std::string model_path = file_directory + "../jit/aimnet2/aimnet2-wb97m-nocut.pt";
    if (not std::ifstream(model_path).good()) {
        SKIP(
            "AIMNet2 model not found (" << model_path << "): configure with"
            << " -DJIT_COMPILE_AIMNET2=ON or run python jit/jit.py --aimnet2"
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
            size, atomic_numbers, no_eval_atoms, "aimnet2-wb97m-nocut", 0, -1,
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
                FLOAT_EQ(vjp_forces[j][c], expect, false);
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
            size, atomic_numbers, no_eval_atoms, "aimnet2-wb97m-nocut", 0, -1,
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
