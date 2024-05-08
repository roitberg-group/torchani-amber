#include <cstdlib>
#include <iostream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/autograd/autograd.h>  // For forces
#include <torch/script.h>

#include <torchani.h>

// Globals:
namespace {
int cached_torchani_model_index = 0;
torch::jit::Module model;
torch::Tensor torchani_atomic_numbers;
// This factor should come straight from torchani.units and be consistent with ASE
double HARTREE_TO_KCALMOL = 627.5094738898777;
// Default device is CPU
torch::Device torchani_device{torch::kCPU, -1};
// Amber has PBC enabled in all directions for PME
torch::Tensor use_pbc = torch::tensor(
    {true, true, true},
    torch::TensorOptions()
        .dtype(torch::kBool)
        .device(torch::Device(torch::kCPU, -1))
);
torch::ScalarType torchani_precision = torch::kFloat;
std::unordered_map<int, std::string> torchani_model = {
    {-1, "custom"},
    {0, "ani1x"},
    {1, "ani1ccx"},
    {2, "ani2x"},
    {3, "animbis"},
    {4, "anidr"}
};
}  // namespace

/**
 * Sets the global torchani device
 * */
void torchani_set_device(bool use_cuda_device, int device_index) {
    if (use_cuda_device) {
        torchani_device = torch::Device(torch::kCUDA, device_index);
    } else {
        // CPU device should always be -1
        if (device_index != -1) {
            std::cerr << "Error in libtorchani\n"
                      << "Device index should be -1 for CPU" << std::endl;
            std::exit(2);
        }
        torchani_device = torch::Device(torch::kCPU, device_index);
    }
}

/**
 * Sets the global torchani precision
 * */
void torchani_set_precision(bool use_double_precision) {
    if (use_double_precision) {
        torchani_precision = torch::kDouble;
    } else {
        torchani_precision = torch::kFloat;
    }
}

// Call with (atomic_numbers, &num_atoms_raw, &use_cuda_device_raw, ...)
void torchani_init_atom_types_(
    int atomic_numbers[],
    int* num_atoms_raw,
    int* device_index_raw,
    int* torchani_model_index_raw,
    int* network_index_raw,
    int* use_double_precision_raw,
    int* use_cuda_device_raw,
    int* use_torch_cell_list_raw,
    int* use_external_neighborlist_raw,
    int* use_cuaev_raw
) {
    // use_cuda_device and use_double_precision should be a bool but it is
    // set to an int pointer for compatibility with C / Fortran
    // torchani_model should be a string but it is set to an int for compatibility
    // with C
    cached_torchani_model_index = *torchani_model_index_raw;
    int num_atoms = *num_atoms_raw;
    int device_index = *device_index_raw;
    int network_index = *network_index_raw;
    bool use_external_neighborlist = static_cast<bool>(*use_external_neighborlist_raw);
    bool use_cell_list = static_cast<bool>(*use_torch_cell_list_raw);
    bool use_cuaev = static_cast<bool>(*use_cuaev_raw);
    bool use_cuda_device = static_cast<bool>(*use_cuda_device_raw);
    bool use_double_precision = static_cast<bool>(*use_double_precision_raw);
    std::string model_jit_fname;
    std::string suffix = "standard";

    if (use_cuaev and not use_cuda_device) {
        std::cerr
            << "Error in libtorchani\n"
            << "A CUDA capable device should be selected to use the cuaev extension"
            << std::endl;
        std::exit(2);
    }

    if (use_cell_list and not use_cuaev) {
        suffix = "-torch-cell-list";
    } else if (use_external_neighborlist and not use_cuaev) {
        suffix = "-external-cell-list";
    } else if (use_cell_list and use_cuaev) {
        suffix = "-cuaev-torch-cell-list";
    } else if (use_external_neighborlist and use_cuaev) {
        suffix = "-cuaev-external-cell-list";
    } else {
        suffix = "-standard";
    }
    if (network_index == -1) {
        model_jit_fname = torchani_model[*torchani_model_index_raw] + suffix + ".pt";
    } else {
        model_jit_fname = torchani_model[*torchani_model_index_raw] + suffix + "-" +
            std::to_string(network_index) + ".pt";
    }

    torchani_set_device(use_cuda_device, device_index);
    torchani_set_precision(use_double_precision);
    // It is VERY important to get types correctly here. If the
    // types don't match then from_blob will not interpret the pointers correctly
    // and as a consequence the tensor will have junk memory inside. If the blob
    // is int then the tensor should be torch::kInt. If the blob is double
    // then the tensor should be torch::kDouble
    //
    // PBC tensor is initialized even if pbc is not needed afterwards,
    // since the tensor occupies almost no space
    use_pbc = torch::tensor(
        {true, true, true},
        torch::TensorOptions().dtype(torch::kBool).device(torchani_device)
    );
    std::string file_path = __FILE__;
    file_path = file_path.substr(0, file_path.find_last_of("/"));
    file_path = file_path.substr(0, file_path.find_last_of("/"));
    std::string jit_model_path = file_path + "/jit/" + model_jit_fname;
    torchani_atomic_numbers = torch::from_blob(
        atomic_numbers, {num_atoms}, torch::TensorOptions().dtype(torch::kInt)
    );
    torchani_atomic_numbers = torchani_atomic_numbers.to(torchani_device);
    torchani_atomic_numbers = torchani_atomic_numbers.to(torch::kLong);
    // This is necessary since torch has to use this tensor to index internally,
    // and only long tensors can be used for this, so widening of the ints has
    // to be performed.
    // Also, torchani needs an extra dimension as batch dimension
    torchani_atomic_numbers = torchani_atomic_numbers.unsqueeze(0);

    // Disable TF32 and FP16 for accuracy
    torch::jit::setGraphExecutorOptimize(false);
    torch::globalContext().setAllowTF32CuBLAS(false);
    torch::globalContext().setAllowTF32CuDNN(false);
    torch::globalContext().setAllowFP16ReductionCuBLAS(false);
    // The model is loaded from a JIT compiled file always.
    try {
        model = torch::jit::load(jit_model_path, torchani_device);
    } catch (const c10::Error& e) {
        std::cerr << "Error in libtorchani\n"
                  << "Could not load model correctly from path: " << jit_model_path
                  << std::endl;
        std::exit(2);
    }
    // This is only necessary for double precision, since
    // the buffers / parameters are kFloat by default
    model.to(torchani_precision);
    // Long tensors are recast to kLong, this is necessary because
    // to(torch::kDouble) also casts buffers to double (or float)
    model.get_method("_recast_long_buffers")({});
}

std::vector<torch::jit::IValue> setup_inputs_pbc_external_neighborlist(
    torch::Tensor& coordinates, torch::Tensor& neighborlist, torch::Tensor& shifts
) {
    std::tuple<torch::Tensor, torch::Tensor> input_tuple = {
        torchani_atomic_numbers, coordinates
    };
    return {input_tuple, neighborlist, shifts};
}

std::vector<torch::jit::IValue> setup_inputs_pbc(
    double pbc_box[][3], torch::Tensor& coordinates
) {
    // uses two "globals", use_pbc and torchani_atomic_numbers
    // Amber's ucell is read into cell
    auto cell_options = torch::TensorOptions().dtype(torch::kDouble);
    torch::Tensor cell = torch::from_blob(pbc_box, {3, 3}, cell_options);
    cell = cell.to(torchani_device);

#if defined(DEBUG)
    torch::Tensor cpu_cell = torch::from_blob(pbc_box, {3, 3}, cell_options);
    cpu_cell = cpu_cell.to(torch::kCPU);
    auto cell_a = cpu_cell.accessor<double, 2>();
    std::cout << "Unit Cell recieved (columns are cell vectors)" << '\n'
              << cell_a[0][0] << " " << cell_a[0][1] << " " << cell_a[0][2] << '\n'
              << cell_a[1][0] << " " << cell_a[1][1] << " " << cell_a[1][2] << '\n'
              << cell_a[2][0] << " " << cell_a[2][1] << " " << cell_a[2][2] << '\n'
              << "Torch output:" << '\n'
              << cpu_cell << '\n'
              << "Transposed torch output" << '\n'
              << torch::transpose(cpu_cell, 0, 1) << std::endl;
#endif

    coordinates = coordinates.to(torchani_precision);
    cell = cell.to(torchani_precision);
    // cell needs to be transposed due to fortran using column major order
    cell = torch::transpose(cell, 0, 1);
    std::tuple<torch::Tensor, torch::Tensor> input_tuple = {
        torchani_atomic_numbers, coordinates
    };
    // Create a vector of input values, jit::script::Module
    // classes accept and return values of ONLY type torch::jit::IValue so
    // any tensor passed to them has to be wrapped in this
    // return by value is OK since inputs is locally created
    return {input_tuple, cell, use_pbc};
}

std::vector<torch::jit::IValue> setup_inputs_nopbc(torch::Tensor& coordinates) {
    // uses one "global", torchani_atomic_numbers
    coordinates = coordinates.to(torchani_precision);
    std::tuple<torch::Tensor, torch::Tensor> input_tuple = {
        torchani_atomic_numbers, coordinates
    };
    // Create a vector of input values, jit::script::Module
    // classes accept and return values of ONLY type torch::jit::IValue so
    // any tensor passed to them has to be wrapped in this
    std::vector<torch::jit::IValue> inputs = {input_tuple};
    // return by value is OK since inputs is locally created
    return inputs;
}

torch::Tensor setup_coordinates(double coordinates_raw[][3], int num_atoms) {
    // Torch needs as an input a TensorDataContainer to the
    // torch::tensor constructor. The TensorDataContainer supports an initializer
    // list, an at::ArrayRef or an std::vector of supported data types.
    // An alternative used here to initialize from a pointer to a sequential
    // memory location that has some data (from_blob).
    // WARNING: It is crucial that
    // the pointer type and the tensor type are the same when doing this,
    // otherwise memory alignment issues occurr
    // coordinates must be set to require grad, since automatic differentiation is
    // used to get the forces
    torch::Tensor coordinates =
        torch::from_blob(
            coordinates_raw,
            {num_atoms, 3},
            torch::TensorOptions().dtype(torch::kDouble).requires_grad(true)
        )
            .to(torchani_device);
    // torchani needs an extra dimension as "batch dimension"
    coordinates = coordinates.unsqueeze(0);
    return coordinates;
}

torch::Tensor setup_neighborlist(int* neighborlist_raw[], int num_neighbors) {
    auto options = torch::TensorOptions().dtype(torch::kInt);
    torch::Tensor neighborlist =
        torch::from_blob(neighborlist_raw, {2, num_neighbors}, options);
    neighborlist = neighborlist.to(torch::kLong);
    return neighborlist.to(torchani_device);
}

torch::Tensor setup_shifts(double shifts_raw[][3], int num_neighbors) {
    auto options = torch::TensorOptions().dtype(torch::kDouble).requires_grad(true);
    torch::Tensor shifts = torch::from_blob(shifts_raw, {num_neighbors, 3}, options);
    return shifts.to(torchani_device);
}

void calculate_and_populate_charge_derivatives(
    torch::Tensor& coordinates,
    torch::Tensor& atomic_charges_tensor,
    double* atomic_charge_derivatives,
    int num_atoms
) {
    for (int atom_idx = 0; atom_idx != num_atoms; ++atom_idx) {
        torch::Tensor charge_deriv = torch::autograd::grad(
            {atomic_charges_tensor.index({0, atom_idx})},
            {coordinates},
            /* grad_outputs= */ {},
            /* retain_graph= */ true
        )[0];
        charge_deriv = charge_deriv.to(torch::kCPU, torch::kFloat64);
        auto charge_deriv_acc = charge_deriv.accessor<double, 3>();
        for (int atom_subidx = 0; atom_subidx != num_atoms; ++atom_subidx) {
            for (int c = 0; c != 3; ++c) {
                atomic_charge_derivatives
                    [c + atom_subidx * 3 + atom_idx * num_atoms * 3] =
                        charge_deriv_acc[0][atom_subidx][c];
            }
        }
    }
}

void calculate_and_populate_qbc_derivatives(
    torch::Tensor& coordinates,
    torch::Tensor& qbc,
    double qbc_derivatives[][3],
    bool retain_graph,
    int num_atoms
) {
    std::vector<torch::Tensor> qbc_vec = {qbc.sum()};
    std::vector<torch::Tensor> coord_vec = {coordinates};
    torch::Tensor torchani_qbc_derivatives = torch::autograd::grad(
                                                 qbc_vec,
                                                 coord_vec,
                                                 /* grad_outputs= */ {},
                                                 /* retain_graph= */ retain_graph
                                             )[0] *
        HARTREE_TO_KCALMOL;
    torchani_qbc_derivatives = torchani_qbc_derivatives.to(torch::kCPU, torch::kDouble);

    auto torchani_qbc_derivatives_a = torchani_qbc_derivatives.accessor<double, 3>();
    for (int atom = 0; atom != num_atoms; ++atom) {
        for (int c = 0; c != 3; ++c) {
            qbc_derivatives[atom][c] = torchani_qbc_derivatives_a[0][atom][c];
        }
    }
}

void calculate_and_populate_forces(
    torch::Tensor& coordinates,
    torch::Tensor& output,
    double forces[][3],
    bool retain_graph,
    int num_atoms
) {
    std::vector<torch::Tensor> energy_vec = {output.sum()};
    std::vector<torch::Tensor> coord_vec = {coordinates};
    // the output is a vector of tensors, of the same number of elements as
    // coord_vec (in this case only one) so the [0] element is the array of
    // derivatives, and the negative is the force
    torch::Tensor torchani_force = -torch::autograd::grad(
                                       energy_vec,
                                       coord_vec,
                                       /* grad_outputs= */ {},
                                       /* retain_graph= */ retain_graph
                                   )[0] *
        HARTREE_TO_KCALMOL;
    // Accessor is used to access tensor elements, but dimensionality has
    // to be known at compile time. Note that this is a CPU accessor.
    // Packed accessors can't be used to access CUDA tensors outside CUDA kernels,
    // AFIK cuda tensors have to be converted to cpu before access is allowed
    // This cast does nothing if output is already of the required type and
    // device, so there should be no overhead.
    torchani_force = torchani_force.to(torch::kCPU, torch::kDouble);

    auto torchani_force_a = torchani_force.accessor<double, 3>();
    for (int atom = 0; atom != num_atoms; ++atom) {
        for (int c = 0; c != 3; ++c) {
            forces[atom][c] = torchani_force_a[0][atom][c];
        }
    }
}

void populate_atomic_charges(
    torch::Tensor& atomic_charges_tensor, double* atomic_charges, int num_atoms
) {
    atomic_charges_tensor = atomic_charges_tensor.to(torch::kCPU, torch::kDouble);
    auto atomic_charges_accessor = atomic_charges_tensor.accessor<double, 2>();

    for (int atom = 0; atom != num_atoms; ++atom) {
        atomic_charges[atom] = atomic_charges_accessor[0][atom];
    }
}

// This function is actually equivalent to the populate_potential_energy function
void populate_qbc(torch::Tensor& output, double* qbc) {
    output = output.to(torch::kCPU, torch::kDouble);
    *qbc = (*output.data_ptr<double>()) * HARTREE_TO_KCALMOL;
}

void populate_potential_energy(torch::Tensor& output, double* potential_energy) {
    // data is copied here, otherwise errors ocurr, since the data dies as soon
    // as the function exits.
    // This cast does nothing if output is already of the required type and
    // device, so there should be no overhead
    output = output.to(torch::kCPU, torch::kDouble);
    *potential_energy = (*output.data_ptr<double>()) * HARTREE_TO_KCALMOL;
}

torch::Tensor get_energy_output(std::vector<torch::jit::IValue>& inputs) {
    // The output value of the model is of type torch::jit::IValue
    // so it has to be converted to tensor to be correctly used
    // output value is a tuple here
    // The output value of the model is an IValue that has to be cast to a
    // (pointer to a) tuple. The first element is the only important one, which is
    // a 1D tensor that holds the potential energy
    torch::Tensor output = model.forward(inputs).toTuple()->elements()[1].toTensor();
    return output;
}

std::tuple<torch::Tensor, torch::Tensor> get_energy_charges_output(
    std::vector<torch::jit::IValue>& inputs
) {
    auto output = model.forward(inputs).toTuple();
    return {output->elements()[1].toTensor(), output->elements()[2].toTensor()};
}

std::vector<torch::Tensor> get_energy_qbc_output(std::vector<torch::jit::IValue>& inputs
) {
    auto output = model.get_method("energies_qbcs")(inputs).toTuple();
    torch::Tensor energy = output->elements()[1].toTensor();
    torch::Tensor qbc = output->elements()[2].toTensor();
    return {energy, qbc};
}

void torchani_energy_force_external_neighborlist_(
    double coordinates_raw[][3],
    int* num_atoms_raw,
    int* num_neighbors_raw,
    int* neighborlist_raw[],
    double shifts_raw[][3],
    /* outputs */
    double forces[][3],
    double* potential_energy
) {
    int num_atoms = *num_atoms_raw;
    int num_neighbors = *num_neighbors_raw;
    torch::Tensor coordinates = setup_coordinates(coordinates_raw, num_atoms);
    torch::Tensor neighborlist = setup_neighborlist(neighborlist_raw, num_neighbors);
    torch::Tensor shifts = setup_shifts(shifts_raw, num_neighbors);

    // Inputs are setup with PBC
    std::vector<torch::jit::IValue> inputs =
        setup_inputs_pbc_external_neighborlist(coordinates, neighborlist, shifts);
    torch::Tensor output = get_energy_output(inputs);
    calculate_and_populate_forces(coordinates, output, forces, false, num_atoms);
    populate_potential_energy(output, potential_energy);
}

void torchani_energy_force_pbc_(
    double coordinates_raw[][3],
    int* num_atoms_raw,
    double forces[][3],
    double pbc_box[][3],
    double* potential_energy
) {
    // Disable TF32 and FP16 for accuracy
    torch::jit::setGraphExecutorOptimize(false);
    torch::globalContext().setAllowTF32CuBLAS(false);
    torch::globalContext().setAllowTF32CuDNN(false);
    torch::globalContext().setAllowFP16ReductionCuBLAS(false);
    int num_atoms = *num_atoms_raw;
    torch::Tensor coordinates = setup_coordinates(coordinates_raw, num_atoms);
    // Inputs are setup with PBC
    std::vector<torch::jit::IValue> inputs = setup_inputs_pbc(pbc_box, coordinates);
    torch::Tensor output = get_energy_output(inputs);
    calculate_and_populate_forces(coordinates, output, forces, false, num_atoms);
    populate_potential_energy(output, potential_energy);
}

/**
 * This function can only be called with the animbis model
 */
void torchani_energy_force_atomic_charges_(
    double coordinates_raw[][3],
    int* num_atoms_raw,
    int* charges_type_raw,
    double forces[][3],
    double* potential_energy,
    double* atomic_charges
) {
    if (*charges_type_raw != 0) {
        std::cerr << "Error in libtorchani\n"
                  << "Charges type should be 0 (only currently supported value)"
                  << std::endl;
        std::exit(2);
    }
    if (cached_torchani_model_index != 3) {
        std::cerr << "Error in libtorchani\n"
                  << "Torchani model should be animbis (index=3) to calculate charges"
                  << std::endl;
        std::exit(2);
    }
    int num_atoms = *num_atoms_raw;
    torch::Tensor coordinates = setup_coordinates(coordinates_raw, num_atoms);
    std::vector<torch::jit::IValue> inputs = setup_inputs_nopbc(coordinates);
    auto output = get_energy_charges_output(inputs);
    torch::Tensor energy = std::get<0>(output);
    torch::Tensor atomic_charges_tensor = std::get<1>(output);

    calculate_and_populate_forces(coordinates, energy, forces, false, num_atoms);
    populate_potential_energy(energy, potential_energy);
    populate_atomic_charges(atomic_charges_tensor, atomic_charges, num_atoms);
}

void torchani_energy_force_atomic_charges_with_derivatives_(
    double coordinates_raw[][3],
    int* num_atoms_raw,
    int* charges_type_raw,
    /* outputs */
    double forces[][3],
    double* potential_energy,
    double* atomic_charge_derivatives,
    double* atomic_charges
) {
    if (*charges_type_raw != 0) {
        std::cerr << "Error in libtorchani\n"
                  << "Charges type should be 0 (only currently supported value)"
                  << std::endl;
        std::exit(2);
    }
    if (cached_torchani_model_index != 3) {
        std::cerr << "Error in libtorchani\n"
                  << "Torchani model should be animbis (index=3) to calculate charges"
                  << std::endl;
        std::exit(2);
    }
    int num_atoms = *num_atoms_raw;
    torch::Tensor coordinates = setup_coordinates(coordinates_raw, num_atoms);
    std::vector<torch::jit::IValue> inputs = setup_inputs_nopbc(coordinates);
    auto output = get_energy_charges_output(inputs);
    torch::Tensor energy = std::get<0>(output);
    torch::Tensor atomic_charges_tensor = std::get<1>(output);

    calculate_and_populate_forces(coordinates, energy, forces, true, num_atoms);
    calculate_and_populate_charge_derivatives(
        coordinates, atomic_charges_tensor, atomic_charge_derivatives, num_atoms
    );
    populate_potential_energy(energy, potential_energy);
    populate_atomic_charges(atomic_charges_tensor, atomic_charges, num_atoms);
}

void torchani_energy_force_(
    double coordinates_raw[][3],
    int* num_atoms_raw,
    double forces[][3],
    double* potential_energy
) {
    // Disable TF32 and FP16 for accuracy
    torch::jit::setGraphExecutorOptimize(false);
    torch::globalContext().setAllowTF32CuBLAS(false);
    torch::globalContext().setAllowTF32CuDNN(false);
    torch::globalContext().setAllowFP16ReductionCuBLAS(false);
    int num_atoms = *num_atoms_raw;
    torch::Tensor coordinates = setup_coordinates(coordinates_raw, num_atoms);
    std::vector<torch::jit::IValue> inputs = setup_inputs_nopbc(coordinates);
    torch::Tensor output = get_energy_output(inputs);
    calculate_and_populate_forces(coordinates, output, forces, false, num_atoms);
    populate_potential_energy(output, potential_energy);
}

void torchani_energy_force_qbc_(
    double coordinates_raw[][3],
    int* num_atoms_raw,
    double forces[][3],
    double* potential_energy,
    double* qbc
) {
    int num_atoms = *num_atoms_raw;
    torch::Tensor coordinates = setup_coordinates(coordinates_raw, num_atoms);
    std::vector<torch::jit::IValue> inputs = setup_inputs_nopbc(coordinates);
    auto output = get_energy_qbc_output(inputs);
    calculate_and_populate_forces(coordinates, output[0], forces, false, num_atoms);
    populate_potential_energy(output[0], potential_energy);
    populate_qbc(output[1], qbc);
}

void torchani_data_for_monitored_mlmm_(
    double coordinates_raw[][3],
    int* num_atoms_raw,
    int* charges_type_raw,
    /* outputs */
    double forces[][3],
    double* potential_energy,
    double* atomic_charge_derivatives,
    double* atomic_charges,
    double* qbc,
    double qbc_derivatives[][3]
) {
    if (*charges_type_raw != 0) {
        std::cerr << "Error in libtorchani\n"
                  << "Charges type should be 0 (only currently supported value)"
                  << std::endl;
        std::exit(2);
    }
    if (cached_torchani_model_index != 3) {
        std::cerr << "Error in libtorchani\n"
                  << "Torchani model should be animbis (index=3) to calculate charges"
                  << std::endl;
        std::exit(2);
    }
    int num_atoms = *num_atoms_raw;
    torch::Tensor coordinates = setup_coordinates(coordinates_raw, num_atoms);
    std::vector<torch::jit::IValue> inputs = setup_inputs_nopbc(coordinates);
    auto output = get_energy_charges_output(inputs);
    torch::Tensor energy_tensor = std::get<0>(output);
    torch::Tensor atomic_charges_tensor = std::get<1>(output);
    torch::Tensor qbc_tensor = get_energy_qbc_output(inputs)[1];

    calculate_and_populate_forces(coordinates, energy_tensor, forces, true, num_atoms);
    calculate_and_populate_qbc_derivatives(
        coordinates, qbc_tensor, qbc_derivatives, true, num_atoms
    );
    calculate_and_populate_charge_derivatives(
        coordinates, atomic_charges_tensor, atomic_charge_derivatives, num_atoms
    );
    populate_potential_energy(energy_tensor, potential_energy);
    populate_atomic_charges(atomic_charges_tensor, atomic_charges, num_atoms);
    populate_qbc(qbc_tensor, qbc);
}
