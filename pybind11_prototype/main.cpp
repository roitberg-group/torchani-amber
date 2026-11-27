#include <iostream>
#include <string>

#include "python_model.h"

int main(int argc, char** argv) {
    if (argc != 2 and argc != 3) {
        std::cerr << "Usage: run_python_model <model-dir> [torch-compile-mode]\n";
        return 1;
    }
    ModelOptions options;
    if (argc == 3) {
        options.do_torch_compile = true;
        options.torch_compile_mode = argv[2];
    }
    PythonModel model{argv[1], torch::kCPU, torch::kDouble, options};
    torch::Tensor atomic_numbers = torch::tensor({8, 1, 1}, torch::kLong);
    torch::Tensor coords = torch::tensor(
        {{0.0, 0.0, 0.0}, {0.96, 0.0, 0.0}, {-0.24, 0.93, 0.0}}, torch::kDouble
    );
    auto [energy, forces] = model.computeEnergyAndForces(atomic_numbers, coords);
    std::cout << "energy:\n" << energy << "\nforces:\n" << forces << "\n";
    return 0;
}
