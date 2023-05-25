#include <fstream>
#include <iostream>
#include <string>
#include <cstdlib>

#include "catch2.hpp"
// This file must be linked to libtorchani
#include "torchani.h"


static std::string file_path = __FILE__;
static std::string file_directory = file_path.substr(0, file_path.find_last_of("/")) + "/";


TEST_CASE("Energy + Force + Charge CPU, MBIS", "[CPU]") {
    double coordinates[][3] = {{3.0, 3.0, 4.0}, {1.0, 2.0, 1.0}};
    int atomic_numbers[] = {1, 6};
    int size = 2;
    double forces[size][3];
    double atomic_charges[size];
    double potential_energy;
    SECTION("Test on CPU double precision"){
        int use_cuda_device = 0;
        int device_index = -1;
        int use_double_precision = 1;
        int torchani_model_index = 3;  // mbis model
        int network_index = 0;
        int use_cell_list = 0;
        int use_external_neighborlist = 0;
        int charges_type_raw = 0;
        // This is hacky and needs fixing
        std::ifstream infile(file_directory + "test_values_cpu_mbis.txt");
        double value;
        double test_values[9];
        if (!infile){
            std::cerr << "Can't open file with test values." << '\n';
            std::exit(-1);
        }
        for (int j = 0; j != 9; ++j){
            infile >> value;
            test_values[j] = value;
        }
        torchani_init_atom_types_(
            atomic_numbers,
            &size,
            &device_index,
            &torchani_model_index,
            &network_index,
            &use_double_precision,
            &use_cuda_device,
            &use_cell_list,
            &use_external_neighborlist
        );
        for (long j = 0; j != 10; ++j){
            torchani_energy_force_atomic_charges_(
                coordinates,
                &size,
                &charges_type_raw,
                /* outputs */
                forces,
                &potential_energy,
                atomic_charges
            );
            CHECK(potential_energy == Approx(test_values[0]));
            CHECK(forces[0][0] == Approx(test_values[1]));
            CHECK(forces[0][1] == Approx(test_values[2]));
            CHECK(forces[0][2] == Approx(test_values[3]));
            CHECK(forces[1][0] == Approx(test_values[4]));
            CHECK(forces[1][1] == Approx(test_values[5]));
            CHECK(forces[1][2] == Approx(test_values[6]));
            CHECK(atomic_charges[0] == Approx(test_values[7]));
            REQUIRE(atomic_charges[1] == Approx(test_values[8]));
        }
    }
}
