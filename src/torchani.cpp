#include <stdint.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/autograd/autograd.h>  // For forces
#include <torch/script.h>

#include <torchani.h>

// Globals:
namespace {
std::string cached_torchani_model = "ani1x";
torch::jit::Module model;
torch::Tensor torchani_atomic_numbers;
// This factor should come straight from torchani.units and be consistent with ASE
double HARTREE_TO_KCALMOL = 627.5094738898777;
// Default device is CPU
torch::Device torchani_device{torch::kCPU, -1};
// Amber has PBC enabled in all directions for PME
torch::Tensor pbc_true = torch::tensor(
    {true, true, true},
    torch::TensorOptions()
        .dtype(torch::kBool)
        .device(torch::Device(torch::kCPU, -1))
);
torch::ScalarType torchani_precision = torch::kFloat;
std::vector<std::string> torchani_builtin_models = {
    "ani1x",
    "ani1ccx",
    "ani2x",
    "animbis",
    "anidr",
    "aniala"
};
}  // namespace

/**
 * Sets the global torchani device
 * */
void torchani_set_device(bool use_cuda_device, int device_index) {
    if (use_cuda_device) {
        torchani_device = torch::Device(torch::kCUDA, device_index);
    } else {
        torchani_device = torch::Device(torch::kCPU, device_index);
    }
    pbc_true = pbc_true.to(torchani_device);
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

torch::Tensor setup_cell(double cell_buf[][3]) {
    auto cell_options = torch::TensorOptions().dtype(torch::kDouble);
    torch::Tensor cell = torch::from_blob(cell_buf, {3, 3}, cell_options);
    return cell.to(torch::TensorOptions().dtype(torchani_precision).device(torchani_device));
}

std::vector<torch::jit::IValue> setup_inputs_pbc(
    double cell_buf[][3], torch::Tensor& coords
) {
    // uses two "globals", pbc_true and torchani_atomic_numbers
    // Amber's ucell is read into cell
    auto cell_options = torch::TensorOptions().dtype(torch::kDouble);
    torch::Tensor cell = torch::from_blob(cell_buf, {3, 3}, cell_options);
    cell = cell.to(torchani_device);

#ifdef DEBUG
    torch::Tensor cpu_cell = torch::from_blob(cell_buf, {3, 3}, cell_options);
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

    coords = coords.to(torchani_precision);
    cell = cell.to(torchani_precision);
    // Cell needs to be transposed due to ANI using opposite cell convention
    cell = torch::transpose(cell, 0, 1);
    std::tuple<torch::Tensor, torch::Tensor> input_tuple = {
        torchani_atomic_numbers, coords
    };
    // Create a vector of input values, jit::script::Module
    // classes accept and return values of ONLY type torch::jit::IValue so
    // any tensor passed to them has to be wrapped in this
    // return by value is OK since inputs is locally created
    return {input_tuple, cell, pbc_true};
}

std::vector<torch::jit::IValue> setup_inputs_nopbc(torch::Tensor& coords) {
    // uses one "global", torchani_atomic_numbers
    coords = coords.to(torchani_precision);
    std::tuple<torch::Tensor, torch::Tensor> input_tuple = {
        torchani_atomic_numbers, coords
    };
    // Create a vector of input values, jit::script::Module
    // classes accept and return values of ONLY type torch::jit::IValue so
    // any tensor passed to them has to be wrapped in this
    std::vector<torch::jit::IValue> inputs = {input_tuple};
    // return by value is OK since inputs is locally created
    return inputs;
}

torch::Tensor setup_coords(int num_atoms, double coords_buf[][3]) {
    // Torch needs as an input a TensorDataContainer to the
    // torch::tensor constructor. The TensorDataContainer supports an initializer
    // list, an at::ArrayRef or an std::vector of supported data types.
    // An alternative used here to initialize from a pointer to a sequential
    // memory location that has some data (from_blob).
    // WARNING: It is crucial that
    // the pointer type and the tensor type are the same when doing this,
    // otherwise memory alignment issues occurr
    // coords must be set to require grad, since automatic differentiation is
    // used to get the forces
    torch::Tensor coords =
        torch::from_blob(
            coords_buf,
            {num_atoms, 3},
            torch::TensorOptions().dtype(torch::kDouble).requires_grad(true)
        );
    // torchani needs an extra dimension as "batch dimension"
    coords = coords.unsqueeze(0);
    return coords.to(torch::TensorOptions().dtype(torchani_precision).device(torchani_device));
}

torch::Tensor setup_neighborlist(int num_neighbors, int* neighborlist_buf[2]) {
    auto options = torch::TensorOptions().dtype(torch::kInt);
    torch::Tensor neighborlist =
        torch::from_blob(neighborlist_buf, {2, num_neighbors}, options);
    return neighborlist.to(torch::TensorOptions().dtype(torch::kLong).device(torchani_device));
}

torch::Tensor setup_shifts(int num_neighbors, double shifts_buf[][3]) {
    auto options = torch::TensorOptions().dtype(torch::kDouble).requires_grad(true);
    torch::Tensor shifts = torch::from_blob(shifts_buf, {num_neighbors, 3}, options);
    return shifts.to(torch::TensorOptions().dtype(torchani_precision).device(torchani_device));
}

void calculate_and_populate_charge_derivatives(
    torch::Tensor& coords,
    torch::Tensor& atomic_charges_tensor,
    double* atomic_charge_derivatives,
    int num_atoms
) {
    for (int atom_idx = 0; atom_idx != num_atoms; ++atom_idx) {
        torch::Tensor charge_deriv = torch::autograd::grad(
            {atomic_charges_tensor.index({0, atom_idx})},
            {coords},
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
    torch::Tensor& coords,
    torch::Tensor& qbc,
    double qbc_derivatives[][3],
    bool retain_graph,
    int num_atoms
) {
    std::vector<torch::Tensor> qbc_vec = {qbc.sum()};
    std::vector<torch::Tensor> coord_vec = {coords};
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
    torch::Tensor& coords,
    torch::Tensor& output,
    double forces_buf[][3],
    bool retain_graph,
    int num_atoms
) {
    std::vector<torch::Tensor> energy_vec = {output.sum()};
    std::vector<torch::Tensor> coord_vec = {coords};
    // the output is a vector of tensors, of the same number of elements as
    // coord_vec (in this case only one) so the [0] element is the array of
    // derivatives, and the negative is the force
    torch::Tensor torchani_force = -torch::autograd::grad(
                                       energy_vec,
                                       coord_vec,
                                       /* grad_outputs= */ {},
                                       /* retain_graph= */ retain_graph
                                   )[0] * HARTREE_TO_KCALMOL;
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
            forces_buf[atom][c] = torchani_force_a[0][atom][c];
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


torch::Tensor get_energy_output_from_external_neighbors(std::vector<torch::jit::IValue>& inputs) {
    // The output value of the model is of type torch::jit::IValue
    // so it has to be converted to tensor to be correctly used
    // output value is a tuple here
    // The output value of the model is an IValue that has to be cast to a
    // (pointer to a) tuple. The first element is the only important one, which is
    // a 1D tensor that holds the potential energy
    torch::Tensor output = model.get_method("compute_from_external_neighbors")(inputs).toTensor();
    return output;
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
    auto output = model.get_method("energies_and_atomic_charges")(inputs).toTuple();
    return {output->elements()[1].toTensor(), output->elements()[2].toTensor()};
}

std::vector<torch::Tensor> get_energy_qbc_output(std::vector<torch::jit::IValue>& inputs
) {
    auto output = model.get_method("energies_qbcs")(inputs).toTuple();
    torch::Tensor energy = output->elements()[1].toTensor();
    torch::Tensor qbc = output->elements()[2].toTensor();
    return {energy, qbc};
}

extern "C" {
void torchani_init_model(
    int num_atoms,
    int atomic_nums[],
    const char* model_type,
    int device_index,
    int network_index,
    bool use_double_precision,
    bool use_cuda_device,
    bool use_cuaev
) {
    cached_torchani_model = model_type;
    std::string model_jit_fname;

    if (use_cuaev and not use_cuda_device) {
        std::cerr
            << "Error in libtorchani\n"
            << "A CUDA capable device must be selected to use the cuAEV extension"
            << std::endl;
        std::exit(2);
    }

    model_jit_fname = cached_torchani_model;
    if (std::find(torchani_builtin_models.begin(), torchani_builtin_models.end(), cached_torchani_model) != torchani_builtin_models.end()) {
        model_jit_fname = model_jit_fname + ".pt";
    }
    #ifdef DEBUG
    std::cout << "model_jit_fname: " << model_jit_fname << '\n';
    #endif

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
    std::string file_path = __FILE__;
    file_path = file_path.substr(0, file_path.find_last_of("/"));
    file_path = file_path.substr(0, file_path.find_last_of("/"));
    std::string jit_model_path = file_path + "/jit/" + model_jit_fname;
    torchani_atomic_numbers = torch::from_blob(
        atomic_nums, {num_atoms}, torch::TensorOptions().dtype(torch::kInt)
    );
    #ifdef DEBUG
    std::cout << "jit_model_path: " << jit_model_path << '\n';
    #endif
    torchani_atomic_numbers = torchani_atomic_numbers.to(torchani_device);
    torchani_atomic_numbers = torchani_atomic_numbers.to(torch::kLong);
    // This is necessary since torch has to use this tensor to index internally,
    // and only long tensors can be used for this, so widening of the ints has
    // to be performed.
    // Also, torchani needs an extra dimension as batch dimension
    torchani_atomic_numbers = torchani_atomic_numbers.unsqueeze(0);
    #ifdef DEBUG
    std::cout << "Initialized torchani with atomic numbers:" << '\n';
    std::cout << torchani_atomic_numbers << '\n';
    #endif

    // Disable TF32 and FP16 for accuracy
    torch::jit::setGraphExecutorOptimize(false);
    torch::globalContext().setAllowTF32CuBLAS(false);
    torch::globalContext().setAllowTF32CuDNN(false);
    torch::globalContext().setAllowFP16ReductionCuBLAS(false);

    #ifdef DEBUG
    std::cout << "Disabled JIT optimizations" << '\n';
    #endif

    // The model is loaded from a JIT compiled file always.
    try {
        model = torch::jit::load(jit_model_path, torchani_device);
    } catch (const c10::Error& e) {
        std::cerr << "Error in libtorchani\n"
                  << "Could not load model correctly from path: " << jit_model_path
                  << std::endl;
        std::exit(2);
    }

    // Set the correct model configuration
    if (network_index != -1) {
        model.get_method("set_active_members")({torch::List<std::int64_t>{network_index}});
    }
    if (use_cuaev) {
        model.get_method("set_strategy")({"cuaev"});
    } else {
        model.get_method("set_strategy")({"pyaev"});
    }
    #ifdef DEBUG
    std::cout << "Loaded JIT-compiled model" << '\n';
    #endif

    // This is only necessary for double precision, since
    // the buffers / parameters are kFloat by default
    model.to(torchani_precision);
    #ifdef DEBUG
    std::cout << "Cast model to the specified precision" << '\n';
    std::cout << "Finalized TorchANI Initialization" << '\n';
    #endif
}

void torchani_energy_force_from_external_neighbors(
    int num_atoms,
    int num_neighbors,
    double coords_buf[][3],
    int* neighborlist_buf[2],
    double shifts_buf[][3],
    /* outputs */
    double forces_buf[][3],
    double* potential_energy_buf
) {
    torch::Tensor coords = setup_coords(num_atoms, coords_buf);
    torch::Tensor neighborlist = setup_neighborlist(num_neighbors, neighborlist_buf);
    torch::Tensor shifts = setup_shifts(num_neighbors, shifts_buf);
    // Cell needs to be transposed due to ANI using opposite cell convention
    std::vector<torch::jit::IValue> inputs{
        torchani_atomic_numbers,
        coords,
        neighborlist,
        shifts,
        /* total_charge= */0,
        /* atomic= */false,
        /* ensemble_values= */false
    };
    torch::Tensor result = get_energy_output_from_external_neighbors(inputs);
    calculate_and_populate_forces(coords, result, forces_buf, false, num_atoms);
    populate_potential_energy(result, potential_energy_buf);
}

void torchani_energy_force_pbc(
    int num_atoms,
    double coords_buf[][3],
    double cell_buf[][3],
    double forces_buf[][3],
    double* potential_energy
) {
    // Disable TF32 and FP16 for accuracy
    torch::jit::setGraphExecutorOptimize(false);
    torch::globalContext().setAllowTF32CuBLAS(false);
    torch::globalContext().setAllowTF32CuDNN(false);
    torch::globalContext().setAllowFP16ReductionCuBLAS(false);
    torch::Tensor coords = setup_coords(num_atoms, coords_buf);
    // Inputs are setup with PBC
    std::vector<torch::jit::IValue> inputs = setup_inputs_pbc(cell_buf, coords);
    torch::Tensor output = get_energy_output(inputs);
    calculate_and_populate_forces(coords, output, forces_buf, false, num_atoms);
    populate_potential_energy(output, potential_energy);
}

void torchani_energy_force_atomic_charges(
    int num_atoms,
    double coords_buf[][3],
    double forces_buf[][3],
    double atomic_charges[],
    double* potential_energy
) {
    if (cached_torchani_model != "animbis") {
        std::cerr << "Error in libtorchani\n"
                  << "Torchani model should be animbis (index=3) to calculate charges"
                  << std::endl;
        std::exit(2);
    }
    torch::Tensor coords = setup_coords(num_atoms, coords_buf);
    std::vector<torch::jit::IValue> inputs = setup_inputs_nopbc(coords);
    auto output = get_energy_charges_output(inputs);
    torch::Tensor energy = std::get<0>(output);
    torch::Tensor atomic_charges_tensor = std::get<1>(output);

    calculate_and_populate_forces(coords, energy, forces_buf, false, num_atoms);
    populate_potential_energy(energy, potential_energy);
    populate_atomic_charges(atomic_charges_tensor, atomic_charges, num_atoms);
}

void torchani_energy_force_atomic_charges_with_derivatives(
    int num_atoms,
    double coords_buf[][3],
    /* outputs */
    double forces_buf[][3],
    double atomic_charges[],
    double* atomic_charge_derivatives,
    double* potential_energy
) {
    if (cached_torchani_model != "animbis") {
        std::cerr << "Error in libtorchani\n"
                  << "Torchani model should be animbis (index=3) to calculate charges"
                  << std::endl;
        std::exit(2);
    }
    torch::Tensor coords = setup_coords(num_atoms, coords_buf);
    std::vector<torch::jit::IValue> inputs = setup_inputs_nopbc(coords);
    auto output = get_energy_charges_output(inputs);
    torch::Tensor energy = std::get<0>(output);
    torch::Tensor atomic_charges_tensor = std::get<1>(output);

    calculate_and_populate_forces(coords, energy, forces_buf, true, num_atoms);
    calculate_and_populate_charge_derivatives(
        coords, atomic_charges_tensor, atomic_charge_derivatives, num_atoms
    );
    populate_potential_energy(energy, potential_energy);
    populate_atomic_charges(atomic_charges_tensor, atomic_charges, num_atoms);
}

void torchani_energy_force(
    int num_atoms,
    double coords_buf[][3],
    double forces_buf[][3],
    double* potential_energy
) {
    // Disable TF32 and FP16 for accuracy
    torch::jit::setGraphExecutorOptimize(false);
    torch::globalContext().setAllowTF32CuBLAS(false);
    torch::globalContext().setAllowTF32CuDNN(false);
    torch::globalContext().setAllowFP16ReductionCuBLAS(false);
    torch::Tensor coords = setup_coords(num_atoms, coords_buf);
    std::vector<torch::jit::IValue> inputs = setup_inputs_nopbc(coords);
    torch::Tensor output = get_energy_output(inputs);
    calculate_and_populate_forces(coords, output, forces_buf, false, num_atoms);
    populate_potential_energy(output, potential_energy);
}

void torchani_energy_force_qbc(
    int num_atoms,
    double coords_buf[][3],
    double forces_buf[][3],
    double* qbc,
    double* potential_energy
) {
    torch::Tensor coords = setup_coords(num_atoms, coords_buf);
    std::vector<torch::jit::IValue> inputs = setup_inputs_nopbc(coords);
    auto output = get_energy_qbc_output(inputs);
    calculate_and_populate_forces(coords, output[0], forces_buf, false, num_atoms);
    populate_potential_energy(output[0], potential_energy);
    populate_qbc(output[1], qbc);
}

void torchani_data_for_monitored_mlmm(
    int num_atoms,
    double coords_buf[][3],
    /* outputs */
    double forces_buf[][3],
    double atomic_charges[],
    /* TODO not sure how to specify type of atomic_charges_grad, besides "buf" */
    double* atomic_charge_derivatives,
    double* qbc,
    double qbc_derivatives[][3],
    double* potential_energy
) {
    torch::Tensor coords = setup_coords(num_atoms, coords_buf);
    std::vector<torch::jit::IValue> inputs = setup_inputs_nopbc(coords);

    torch::Tensor energy_tensor = torch::empty(0);
    torch::Tensor atomic_charges_tensor = torch::empty(0);
    if (cached_torchani_model != "animbis") {
        std::cerr << "Error in libtorchani\n"
                  << "Torchani model should be animbis (index=3)"
                  << std::endl;
        std::exit(2);
    }
    auto output = get_energy_charges_output(inputs);
    energy_tensor = std::get<0>(output);
    atomic_charges_tensor = std::get<1>(output);
    torch::Tensor qbc_tensor = get_energy_qbc_output(inputs)[1];

    calculate_and_populate_forces(coords, energy_tensor, forces_buf, true, num_atoms);
    calculate_and_populate_qbc_derivatives(
        coords, qbc_tensor, qbc_derivatives, true, num_atoms
    );
    populate_potential_energy(energy_tensor, potential_energy);
    populate_qbc(qbc_tensor, qbc);
    calculate_and_populate_charge_derivatives(
        coords, atomic_charges_tensor, atomic_charge_derivatives, num_atoms
    );
    populate_atomic_charges(atomic_charges_tensor, atomic_charges, num_atoms);
}
}
