#include <fstream>
#include <iostream>
#include <string>
#include <cstdlib>

#include "catch2.hpp"
// This file must be linked to libtorchani
#include "torchani.h"


static std::string file_path = __FILE__;
static std::string file_directory = file_path.substr(0, file_path.find_last_of("/")) + "/";


TEST_CASE("Call energy force CPU", "[CPU]"){
    double coordinates[][3] = {{3.0, 3.0, 4.0}, {1.0, 2.0, 1.0}};
    int atomic_numbers[] = {1, 6};
    int size = 2;
    double forces[size][3];
    double cell[3][3] = {{10., 0., 0.}, {0., 10., 0.}, {0., 0., 10.}};
    double potential_energy;
    SECTION("Test on CPU double precision"){
        int use_gpu = 0;
        int device_index = -1;
        int use_double_precision = 1;
        int model_type = 0;
        int model_index = 0;
        int use_cell_list = 0;
        int use_external_neighborlist = 0;
        // This is hacky and needs fixing
        std::ifstream infile(file_directory + "test_values_cpu.txt");
        double value;
        double test_values[7];
        if (!infile){
            std::cerr << "Can't open file with test values." << '\n';
            std::exit(-1);
        }
        for (int j = 0; j != 8; ++j){
            infile >> value;
            test_values[j] = value;
        }
        torchani_init_atom_types_(
            atomic_numbers,
            &size,
            &use_gpu,
            &device_index,
            &use_double_precision,
            &model_type,
            &model_index,
            &use_cell_list,
            &use_external_neighborlist
        );
        for (long j = 0; j != 10; ++j){
            torchani_energy_force_pbc_(
                coordinates,
                &size,
                forces,
                cell,
                &potential_energy
            );
            CHECK(potential_energy == Approx(test_values[0]));
            CHECK(forces[0][0] == Approx(test_values[1]));
            CHECK(forces[0][1] == Approx(test_values[2]));
            CHECK(forces[0][2] == Approx(test_values[3]));
            CHECK(forces[1][0] == Approx(test_values[4]));
            CHECK(forces[1][1] == Approx(test_values[5]));
            REQUIRE(forces[1][2] == Approx(test_values[6]));
        }
    }
    SECTION("Test on CPU single precision"){
        int use_gpu = 0;
        int device_index = -1;
        int use_double_precision = 0;
        int model_type = 0;
        int model_index = 0;
        int use_cell_list = 0;
        int use_external_neighborlist = 0;
        // This is hacky and needs fixing
        std::ifstream infile(file_directory + "test_values_cpu.txt");
        double value;
        double test_values[7];
        if (!infile){
            std::cerr << "Can't open file with test values." << '\n';
            std::exit(-1);
        }
        for (int j = 0; j != 8; ++j){
            infile >> value;
            test_values[j] = value;
        }
        torchani_init_atom_types_(
            atomic_numbers,
            &size,
            &use_gpu,
            &device_index,
            &use_double_precision,
            &model_type,
            &model_index,
            &use_cell_list,
            &use_external_neighborlist
        );
        for (long j = 0; j != 10; ++j){
            torchani_energy_force_pbc_(
                coordinates,
                &size,
                forces,
                cell,
                &potential_energy
            );
            CHECK(potential_energy == Approx(test_values[0]));
            CHECK(forces[0][0] == Approx(test_values[1]));
            CHECK(forces[0][1] == Approx(test_values[2]));
            CHECK(forces[0][2] == Approx(test_values[3]));
            CHECK(forces[1][0] == Approx(test_values[4]));
            CHECK(forces[1][1] == Approx(test_values[5]));
            REQUIRE(forces[1][2] == Approx(test_values[6]));
        }
    }
}
