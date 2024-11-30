#include <torchani.h>
#include "config.h"
#include "electro.h"
#include "build_tensors.h"

#include <cmath>
#include <chrono>
#include <stdint.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/autograd/autograd.h>  // For forces
#include <torch/script.h>

// Globals:
namespace {
// Default device is CPU, and default kind is Float32
GlobalConfig config{};
std::string cached_torchani_model = "ani1x";
torch::jit::Module model;
torch::Tensor torchani_atomic_numbers;
// This factor should come straight from torchani.units and be consistent with ASE
double HARTREE_TO_KCALMOL = 627.5094738898777;
// This factor is used by ml/mm for now
double CODATA08_ANGSTROM_TO_BOHR = 1.8897261328727875;
std::vector<std::string> torchani_builtin_models = {
    "ani1x", "ani1ccx", "ani2x", "animbis", "anidr", "aniala"
};
}  // namespace

std::vector<torch::jit::IValue> setup_inputs_pbc(
    torch::Tensor& coords, torch::Tensor& cell, bool ensemble_values = false
) {
    // Create a vector of input values, jit::script::Module
    // classes accept and return values of ONLY type torch::jit::IValue so
    // any tensor passed to them has to be wrapped in this
    // return by value is OK since inputs is locally created
    return {
        std::tuple{torchani_atomic_numbers, coords},
        /* cell= */ cell,
        /* pbc= */ config.enabled_pbc(),
        /* charge= */ 0,
        /* atomic= */ false,
        /* ensemble_values= */ ensemble_values
    };
}

std::vector<torch::jit::IValue> setup_inputs_nopbc(
    torch::Tensor& coords, bool ensemble_values = false
) {
    // uses one "global", torchani_atomic_numbers
    // Create a vector of input values, jit::script::Module
    // classes accept and return values of ONLY type torch::jit::IValue so
    // any tensor passed to them has to be wrapped in this
    // return by value is OK since inputs is locally created
    return {
        std::tuple{torchani_atomic_numbers, coords},
        /* cell= */ torch::indexing::None,
        /* pbc= */ torch::indexing::None,
        /* charge= */ 0,
        /* atomic= */ false,
        /* ensemble_values= */ ensemble_values
    };
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

void calculate_and_populate_embedding_forces(
    torch::Tensor energy,
    torch::Tensor coords,
    torch::Tensor env_charges_coords,
    int num_atoms,
    int num_env_charges,
    double forces_on_atoms_buf[][3],
    double forces_on_env_charges_buf[][3]
) {
    std::vector<torch::Tensor> energy_vec = {energy.sum()};
    std::vector<torch::Tensor> coord_vec = {coords, env_charges_coords};
    // the output is a vector of tensors, of the same number of elements as
    // coord_vec (in this case only one) so the [0] element is the array of
    // derivatives, and the negative is the force
    auto ad_output = torch::autograd::grad(
        energy_vec,
        coord_vec,
        /* grad_outputs= */ {},
        /* retain_graph= */ false
    );
    torch::Tensor forces_on_atoms = -ad_output[0] * HARTREE_TO_KCALMOL;
    torch::Tensor forces_on_env_charges = -ad_output[1] * HARTREE_TO_KCALMOL;
    // Accessor is used to access tensor elements, but dimensionality has
    // to be known at compile time. Note that this is a CPU accessor.
    // Packed accessors can't be used to access CUDA tensors outside CUDA kernels,
    // AFIK cuda tensors have to be converted to cpu before access is allowed
    // This cast does nothing if output is already of the required type and
    // device, so there should be no overhead.
    forces_on_atoms = forces_on_atoms.to(torch::kCPU, torch::kDouble);
    auto force_a = forces_on_atoms.accessor<double, 3>();
    for (int atom = 0; atom != num_atoms; ++atom) {
        for (int c = 0; c != 3; ++c) {
            forces_on_atoms_buf[atom][c] = force_a[0][atom][c];
        }
    }
    forces_on_env_charges = forces_on_env_charges.to(torch::kCPU, torch::kDouble);
    auto qforce_a = forces_on_env_charges.accessor<double, 3>();
    for (int q = 0; q != num_env_charges; ++q) {
        for (int c = 0; c != 3; ++c) {
            forces_on_env_charges_buf[q][c] = qforce_a[0][q][c];
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

// The output value of the model is of type torch::jit::IValue
// so it has to be converted to tensor to be correctly used
// output value is a tuple here
// The output value of the model is an IValue that has to be cast to a
// (pointer to a) tuple. The first element is the only important one, which is
// a 1D tensor that holds the potential energy

torch::Tensor calc_qbcs(int num_atoms, torch::Tensor& energy) {
    torch::Tensor qbc = torch::std(energy, 0);
    torch::Tensor num_atoms_tensor = torch::tensor(num_atoms);
    return qbc / config.tensor_to_torchani_dtype_and_device(num_atoms_tensor).sqrt();
}

void validate_model_output(
    const torch::jit::IValue& output, size_t expect_len, bool external = false
) {
    try {
        output.toTuple();
    } catch (const c10::Error& e) {
        std::cerr << "Error in libtorchani\n"
                  << "Expected model to return a tuple of tensors" << std::endl;
        std::exit(2);
    }
    if (output.toTuple()->elements().size() < expect_len) {
        std::cerr << "Error in libtorchani\n";
        std::cerr << "Expected model to return ";
        if (expect_len == 2) {
            if (external) {
                std::cerr << "a tuple (energies, atomic_scalars)";
            } else {
                std::cerr << "a tuple (species, energies, ...)";
            }
        } else if (expect_len == 3) {
            std::cerr << "a tuple (species, energies, atomic_charges, ...)";
        } else {
            std::cerr << "a tuple of len at least " << expect_len;
        }
        std::cerr << std::endl;
        std::exit(2);
    }
}

bool model_has_method(std::string name) {
    try {
        model.get_method(name);
        return true;
    } catch (const c10::Error& e) {
        return false;
    }
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
        std::cerr << "Error in libtorchani\n"
                  << "A CUDA capable device must be selected to use the cuAEV extension"
                  << std::endl;
        std::exit(2);
    }

    model_jit_fname = cached_torchani_model;
    if (std::find(
            torchani_builtin_models.begin(),
            torchani_builtin_models.end(),
            cached_torchani_model
        ) != torchani_builtin_models.end()) {
        model_jit_fname = model_jit_fname + ".pt";
    }
#ifdef DEBUG
    std::cout << "model_jit_fname: " << model_jit_fname << '\n';
#endif

    config.set_device_and_precision(
        use_cuda_device, device_index, use_double_precision
    );
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
    torchani_atomic_numbers = torchani_atomic_numbers.to(config.device());
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
        model = torch::jit::load(jit_model_path, config.device());
    } catch (const c10::Error& e) {
        std::cerr << "Error in libtorchani\n"
                  << "Could not load model correctly from path: " << jit_model_path
                  << std::endl;
        std::exit(2);
    }

    // This is only necessary for double precision, since
    // the buffers / parameters are kFloat by default
    model.to(config.dtype());
#ifdef DEBUG
    std::cout << "Cast model to the specified precision" << '\n';
#endif

    // Set the correct model configuration
    if (network_index != -1) {
        if (model_has_method("set_active_members")) {
            model.get_method("set_active_members")(
                {torch::List<std::int64_t>{network_index}}
            );
        } else {
            std::cerr << "Error in libtorchani\n"
                      << "You set 'network_index' to a value != -1"
                      << " but the selected model doesn't export"
                      << " a method 'set_active_members(str) -> None'" << std::endl;
            std::exit(2);
        }
    }
    if (use_cuaev) {
        if (model_has_method("set_strategy")) {
            model.get_method("set_strategy")({"cuaev"});
        } else {
            std::cerr << "Error in libtorchani\n"
                      << "You set 'use_cuaev=true'"
                      << " but the selected model doesn't export"
                      << " a method 'set_strategy(str) -> None'"
                      << " that accepts the string 'cuaev'." << std::endl;
            std::exit(2);
        }
    } else {
        // It is not required that models support this
        try {
            model.get_method("set_strategy")({"pyaev"});
        } catch (const c10::Error& e) {
        }
    }

#ifdef DEBUG
    std::cout << "Loaded JIT-compiled model" << '\n';
#endif

#ifdef DEBUG
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
#ifdef TIMING
    auto start = std::chrono::high_resolution_clock::now();
#endif
    torch::Tensor coords = dbl_buf_to_coords_tensor(config, coords_buf, num_atoms);
    torch::Tensor shifts = dbl_buf_to_shifts_tensor(config, shifts_buf, num_neighbors);
    torch::Tensor neighborlist =
        int_buf_to_neighborlist_tensor(config, neighborlist_buf, num_neighbors);
    std::vector<torch::jit::IValue> inputs{
        torchani_atomic_numbers,
        coords,
        neighborlist,
        shifts,
        /* total_charge= */ 0,
        /* atomic= */ false,
        /* ensemble_values= */ false
    };
    if (model_has_method("compute_from_external_neighbors")) {
        torch::jit::IValue output =
            model.get_method("compute_from_external_neighbors")(inputs);
        validate_model_output(output, 2, true);
        torch::Tensor energy = output.toTuple()->elements()[0].toTensor();
        calculate_and_populate_forces(coords, energy, forces_buf, false, num_atoms);
        populate_potential_energy(energy, potential_energy_buf);
    } else {
        std::cerr << "Error in libtorchani\n"
                  << "To use an external neighborlist"
                  << " the model must export 'compute_from_external_neighbors(...)'."
                  << " Consult the TorchANI-Amber readmi for more info" << std::endl;
        std::exit(2);
    }
#ifdef TIMING
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout
        << "TORCHANI-AMBER: energy force external neighbors time"
        << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() *
            1000
        << "ms" << std::endl;
#endif
}

void torchani_energy_force_pbc(
    int num_atoms,
    double coords_buf[][3],
    double cell_buf[][3],
    double forces_buf[][3],
    double* potential_energy
) {
#ifdef TIMING
    auto start = std::chrono::high_resolution_clock::now();
#endif
    torch::Tensor coords = dbl_buf_to_coords_tensor(config, coords_buf, num_atoms);
    torch::Tensor cell = dbl_buf_to_cell_tensor(config, cell_buf);
    // Inputs are setup with PBC
    std::vector<torch::jit::IValue> inputs = setup_inputs_pbc(coords, cell);
    torch::jit::IValue output = model.forward(inputs);
    validate_model_output(output, 2);
    torch::Tensor energy = output.toTuple()->elements()[1].toTensor();
    calculate_and_populate_forces(coords, energy, forces_buf, false, num_atoms);
    populate_potential_energy(energy, potential_energy);
#ifdef TIMING
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout
        << "TORCHANI-AMBER: energy force PBC time"
        << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() *
            1000
        << "ms" << std::endl;
#endif
}

void torchani_energy_force_atomic_charges(
    int num_atoms,
    double coords_buf[][3],
    /* outputs */
    double forces_buf[][3],
    double atomic_charges_buf[],
    double* potential_energy
) {
    torch::Tensor coords = dbl_buf_to_coords_tensor(config, coords_buf, num_atoms);
    std::vector<torch::jit::IValue> inputs = setup_inputs_nopbc(coords);
    torch::jit::IValue output = model.forward(inputs);
    validate_model_output(output, 3);
    torch::Tensor energy = output.toTuple()->elements()[1].toTensor();
    torch::Tensor atomic_charges = output.toTuple()->elements()[2].toTensor();

    calculate_and_populate_forces(coords, energy, forces_buf, false, num_atoms);
    populate_potential_energy(energy, potential_energy);
    populate_atomic_charges(atomic_charges, atomic_charges_buf, num_atoms);
}

/**
 * Note that currently this function can't split the QM forces in two parts, it only
 * outputs net forces due to the model in vacuum and the potential energies
 */
void torchani_energy_force_simple_polarizable_embedding(
    int num_atoms,
    int num_env_charges,
    double distortion_k,
    double coords_buf[][3],
    double atomic_alphas_buf[],  // shape (num-atoms,)
    double env_charge_coords_buf[][3],  //  shape (num-charges, 3)
    double env_charges_buf[],  // shape (num-charges,)
    /* outputs */
    double forces_on_atoms_buf[][3],  // shape (num-atoms, 3)
    double forces_on_env_charges_buf[][3],  // shape (num-charges, 3)
    double atomic_charges_buf[],  // shape (num-atoms, 3)
    double* ene_pot_invacuo_buf,
    double* ene_pot_embed_pol_buf,
    double* ene_pot_embed_dist_buf,
    double* ene_pot_embed_coulomb_buf,
    double* ene_pot_total_buf
) {
    torch::Tensor coords = dbl_buf_to_coords_tensor(config, coords_buf, num_atoms);
    std::vector<torch::jit::IValue> inputs = setup_inputs_nopbc(coords);
    torch::jit::IValue output = model.forward(inputs);
    validate_model_output(output, 3);
    torch::Tensor ene_pot_invacuo = output.toTuple()->elements()[1].toTensor();
    torch::Tensor atomic_charges = output.toTuple()->elements()[2].toTensor();

    // Embedding part
    torch::Tensor atomic_alphas =
        dbl_buf_to_tensor(config, atomic_alphas_buf, {num_atoms});
    torch::Tensor env_charge_coords =
        dbl_buf_to_coords_tensor(config, env_charge_coords_buf, num_env_charges);
    torch::Tensor env_charges =
        dbl_buf_to_tensor(config, env_charges_buf, {num_env_charges});

    torch::Tensor delta = torch::reshape(env_charge_coords, {-1, 1, 3}) -
        torch::reshape(coords, {1, -1, 3});
    torch::Tensor env_charges_to_atoms_distances =
        torch::linalg_norm(delta, std::nullopt, 2);

    // Embedding calculations are made in atomic units, so outputs are in Ha
    // TODO: check these units, they are suspicious
    auto ene_pot_coulomb = electro::coulombic_embedding_energy(
        atomic_charges, env_charges, env_charges_to_atoms_distances
    ) / CODATA08_ANGSTROM_TO_BOHR;
    auto ene_pot_pol = electro::polarizable_embedding_energy(
        coords,
        atomic_alphas,
        env_charge_coords,
        env_charges,
        env_charges_to_atoms_distances
    ) / CODATA08_ANGSTROM_TO_BOHR;
    auto ene_pot_dist = -distortion_k * ene_pot_pol;

    auto ene_pot_total = ene_pot_invacuo + ene_pot_dist + ene_pot_pol + ene_pot_coulomb;
    // TODO It may be possible to return forces due to "different things"
    // by detaching the coords before the final calculation

    // TODO: Retain grad in atomic_charges so that the derivatives
    // can be returned. Use backwargs instead of autograd::grad for simplicity to do
    // this maybe
    calculate_and_populate_embedding_forces(
        ene_pot_total,
        coords,
        env_charge_coords,
        num_atoms,
        num_env_charges,
        forces_on_atoms_buf,
        forces_on_env_charges_buf
    );
    populate_potential_energy(ene_pot_total, ene_pot_total_buf);
    populate_potential_energy(ene_pot_invacuo, ene_pot_invacuo_buf);
    populate_potential_energy(ene_pot_pol, ene_pot_embed_pol_buf);
    populate_potential_energy(ene_pot_dist, ene_pot_embed_dist_buf);
    populate_potential_energy(ene_pot_coulomb, ene_pot_embed_coulomb_buf);

    populate_atomic_charges(atomic_charges, atomic_charges_buf, num_atoms);
}

void torchani_energy_force_atomic_charges_with_derivatives(
    int num_atoms,
    double coords_buf[][3],
    /* outputs */
    double forces_buf[][3],
    double atomic_charges_buf[],
    double* atomic_charge_derivatives,
    double* potential_energy
) {
    torch::Tensor coords = dbl_buf_to_coords_tensor(config, coords_buf, num_atoms);
    std::vector<torch::jit::IValue> inputs = setup_inputs_nopbc(coords);
    torch::jit::IValue output = model.forward(inputs);
    validate_model_output(output, 3);
    torch::Tensor energy = output.toTuple()->elements()[1].toTensor();
    torch::Tensor atomic_charges = output.toTuple()->elements()[2].toTensor();

    calculate_and_populate_forces(coords, energy, forces_buf, true, num_atoms);
    calculate_and_populate_charge_derivatives(
        coords, atomic_charges, atomic_charge_derivatives, num_atoms
    );
    populate_potential_energy(energy, potential_energy);
    populate_atomic_charges(atomic_charges, atomic_charges_buf, num_atoms);
}

void torchani_energy_force(
    int num_atoms,
    double coords_buf[][3],
    double forces_buf[][3],
    double* potential_energy
) {
#ifdef TIMING
    auto start = std::chrono::high_resolution_clock::now();
#endif
    torch::Tensor coords = dbl_buf_to_coords_tensor(config, coords_buf, num_atoms);
    std::vector<torch::jit::IValue> inputs = setup_inputs_nopbc(coords);

    torch::jit::IValue output = model.forward(inputs);
    validate_model_output(output, 2);
    torch::Tensor energy = output.toTuple()->elements()[1].toTensor();
    calculate_and_populate_forces(coords, energy, forces_buf, false, num_atoms);
    populate_potential_energy(energy, potential_energy);
#ifdef TIMING
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout
        << "TORCHANI-AMBER: energy force time"
        << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() *
            1000
        << "ms" << std::endl;
#endif
}

void torchani_energy_force_qbc(
    int num_atoms,
    double coords_buf[][3],
    double forces_buf[][3],
    double* qbc_buf,
    double* potential_energy
) {
    torch::Tensor coords = dbl_buf_to_coords_tensor(config, coords_buf, num_atoms);
    std::vector<torch::jit::IValue> inputs =
        setup_inputs_nopbc(coords, /*ensemble_values=*/true);

    torch::jit::IValue output = model.forward(inputs);
    validate_model_output(output, 2);
    torch::Tensor ensemble_energy = output.toTuple()->elements()[1].toTensor();

    torch::Tensor qbc = calc_qbcs(num_atoms, ensemble_energy);
    torch::Tensor energy = ensemble_energy.mean(0);
    calculate_and_populate_forces(coords, energy, forces_buf, false, num_atoms);
    populate_potential_energy(energy, potential_energy);
    populate_qbc(qbc, qbc_buf);
}

void torchani_data_for_monitored_mlmm(
    int num_atoms,
    double coords_buf[][3],
    /* outputs */
    double forces_buf[][3],
    double atomic_charges_buf[],
    /* TODO not sure how to specify type of atomic_charges_grad, besides "buf" */
    double* atomic_charge_derivatives,
    double* qbc_buf,
    double qbc_derivatives[][3],
    double* potential_energy
) {
    torch::Tensor coords = dbl_buf_to_coords_tensor(config, coords_buf, num_atoms);
    std::vector<torch::jit::IValue> inputs =
        setup_inputs_nopbc(coords, /*ensemble_values=*/true);

    torch::jit::IValue output = model.forward(inputs);
    validate_model_output(output, 3);
    torch::Tensor ensemble_energy = output.toTuple()->elements()[1].toTensor();
    // Squeeze ensemble dimension from atomic charges
    torch::Tensor atomic_charges =
        output.toTuple()->elements()[2].toTensor().squeeze(0);
    torch::Tensor qbc = calc_qbcs(num_atoms, ensemble_energy);
    torch::Tensor energy = ensemble_energy.mean(0);

    calculate_and_populate_forces(coords, energy, forces_buf, true, num_atoms);
    calculate_and_populate_qbc_derivatives(
        coords, qbc, qbc_derivatives, true, num_atoms
    );
    populate_potential_energy(energy, potential_energy);
    populate_qbc(qbc, qbc_buf);
    calculate_and_populate_charge_derivatives(
        coords, atomic_charges, atomic_charge_derivatives, num_atoms
    );
    populate_atomic_charges(atomic_charges, atomic_charges_buf, num_atoms);
}
}
