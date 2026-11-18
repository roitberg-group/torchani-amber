#include <torchani.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <ATen/Context.h>
#include <c10/util/Logging.h>
#include <torch/all.h>
#include <torch/csrc/autograd/autograd.h>  // For forces

#include "tensor_utils.h"
#include "config.h"
#include "electro.h"
#include "mlip.h"
#include "timing.h"

// Globals:
namespace {
// Default device is CPU, and default kind is Float32
config::GlobalConfig global_config{};
std::unique_ptr<mlip::MlipModel> model;
torch::Tensor torchani_atomic_numbers;
torch::Tensor torchani_do_eval_atoms_mask;
torch::Tensor torchani_do_eval_idxs;
// Graph kept between torchani_energy_atomic_charges_retain and
// torchani_energy_atomic_charges_vjp
torch::Tensor retained_coords;
torch::Tensor retained_energy;
torch::Tensor retained_atomic_charges;
bool charges_graph_retained = false;
// This factor should come straight from torchani.units and be consistent with ASE
double HARTREE_TO_KCALMOL = 627.5094738898777;
// NOTE: Probably this should be used instead (or nothing)
// double AMBER_HARTREE_TO_KCALMOL = 627.509469433585;
std::vector<std::string> torchani_builtin_models = {
    "ani1x",
    "ani1x-infer",
    "ani1ccx",
    "ani1ccx-infer",
    "ani2x",
    "ani2x-infer",
    "ani2xr",
    "ani2xr-infer",
    "ani2dr",
    "ani2dr-infer",
    "animbis",
    "animbis-infer",
    "animbis-custom-mlmm",
    "anir2s",
    "anir2s-infer",
    "anir2s_water",
    "anir2s_water-infer",
    "anir2s_chcl3",
    "anir2s_chcl3-infer",
    "anir2s_ch3cn",
    "anir2s_ch3cn-infer",
    "aimnet2-b973c-dsf",
    "aimnet2-b973c-ewald",
    "aimnet2-b973c-nocut",
    "aimnet2-b973c-mbis-dsf",
    "aimnet2-b973c-mbis-ewald",
    "aimnet2-b973c-mbis-nocut",
    "aimnet2-wb97m-dsf",
    "aimnet2-wb97m-ewald",
    "aimnet2-wb97m-nocut",
    "aimnet2-wb97m-mbis-dsf",
    "aimnet2-wb97m-mbis-ewald",
    "aimnet2-wb97m-mbis-nocut",
    "nutmeg-small",
    "nutmeg-medium",
    "nutmeg-large",
};
}  // namespace

void populateAtomicChargeDerivatives(
    torch::Tensor atomic_charge_grads,
    double* atomic_charge_derivatives,
    int num_atoms
) {
    atomic_charge_grads = atomic_charge_grads.to(torch::kCPU, torch::kFloat64);
    auto atomic_charge_grads_a = atomic_charge_grads.accessor<double, 3>();
    for (int atom_idx = 0; atom_idx != num_atoms; ++atom_idx) {
        for (int atom_subidx = 0; atom_subidx != num_atoms; ++atom_subidx) {
            for (int c = 0; c != 3; ++c) {
                atomic_charge_derivatives
                    [c + atom_subidx * 3 + atom_idx * num_atoms * 3] =
                        atomic_charge_grads_a[atom_idx][atom_subidx][c];
            }
        }
    }
}

void populateQbcDerivatives(
    torch::Tensor qbc_grads,
    double qbc_derivatives[][3],
    int num_atoms
) {
    qbc_grads = (qbc_grads * HARTREE_TO_KCALMOL).to(torch::kCPU, torch::kDouble);
    auto qbc_grads_a = qbc_grads.accessor<double, 3>();
    for (int atom = 0; atom != num_atoms; ++atom) {
        for (int c = 0; c != 3; ++c) {
            qbc_derivatives[atom][c] = qbc_grads_a[0][atom][c];
        }
    }
}

void populateEmbeddingForces(
    torch::Tensor forces_on_atoms,
    torch::Tensor forces_on_env_charges,
    int num_atoms,
    int num_env_charges,
    double forces_on_atoms_buf[][3],
    double forces_on_env_charges_buf[][3]
) {
    forces_on_atoms = forces_on_atoms * HARTREE_TO_KCALMOL;
    forces_on_env_charges = forces_on_env_charges * HARTREE_TO_KCALMOL;
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

void populateForces(torch::Tensor forces, double forces_buf[][3], int num_atoms) {
    torch::Tensor torchani_force = forces * HARTREE_TO_KCALMOL;
    // Accessor is used to access tensor elements, but dimensionality has
    // to be known at compile time. Note that this is a CPU accessor.
    // Packed accessors can't be used to access CUDA tensors outside CUDA kernels,
    // AFIK cuda tensors have to be converted to cpu before access is allowed
    // This cast does nothing if output is already of the required type and
    // device, so there should be no overhead.
    //
    torchani_force = torchani_force.to(torch::kCPU, torch::kDouble);
    if (global_config.useNoEvalAtoms()) {
        int idx = 0;
        auto do_eval_atoms_mask_clone  = torchani_do_eval_atoms_mask.clone().to(torch::kCPU);
        auto do_eval_atoms_mask_a = do_eval_atoms_mask_clone.accessor<bool, 1>();
        auto torchani_force_a = torchani_force.accessor<double, 3>();
        for (int atom = 0; atom != num_atoms; ++atom) {
            if (do_eval_atoms_mask_a[atom]) {
                for (int c = 0; c != 3; ++c) {
                    forces_buf[atom][c] = torchani_force_a[0][idx][c];
                }
                idx += 1;
            } else {
                for (int c = 0; c != 3; ++c) {
                    forces_buf[atom][c] = 0.0;
                }
            }
        }
    } else {
        auto torchani_force_a = torchani_force.accessor<double, 3>();
        for (int atom = 0; atom != num_atoms; ++atom) {
            for (int c = 0; c != 3; ++c) {
                forces_buf[atom][c] = torchani_force_a[0][atom][c];
            }
        }
    }
}

void populateAtomicCharges(
    torch::Tensor& atomic_charges_tensor, double* atomic_charges, int num_atoms
) {
    atomic_charges_tensor = atomic_charges_tensor.to(torch::kCPU, torch::kDouble);
    auto atomic_charges_accessor = atomic_charges_tensor.accessor<double, 2>();

    for (int atom = 0; atom != num_atoms; ++atom) {
        atomic_charges[atom] = atomic_charges_accessor[0][atom];
    }
}

void populateEnergyScalar(torch::Tensor& output, double* energy_scalar) {
    // data is copied here, otherwise errors ocurr, since the data dies as soon
    // as the function exits.
    // This cast does nothing if output is already of the required type and
    // device, so there should be no overhead
    output = output.to(torch::kCPU, torch::kDouble);
    *energy_scalar = (*output.data_ptr<double>()) * HARTREE_TO_KCALMOL;
}

void exitIfNoEvalAtoms(const config::GlobalConfig& config, const std::string& fn_name) {
    if (config.useNoEvalAtoms()) {
        std::cerr << "ERROR (libtorchani)\n"
                  << fn_name << " does not support"
                  << " atoms excluded from the evaluation" << std::endl;
        std::exit(2);
    }
}

void exitIfNotInitialized(const config::GlobalConfig& config, const std::string& fn_name) {
    if (not config.torchaniIsInit()) {
        std::cerr << "ERROR (libtorchani)\n"
                  << fn_name << " was called before torchani_initialize" << std::endl;
        std::exit(2);
    }
}

void disableTorchReducedFloatPrecisionOptimizations() {
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
    at::globalContext().setAllowFP16ReductionCuBLAS(false);
    LOG(INFO) << "Disabled TF32 in cuBLAS and cuDNN, and FP16 reduction in cuBLAS";
}

extern "C" {
void torchani_initialize(
    int num_atoms,
    int atomic_nums[],
    int no_eval_atoms[],  // Shape (num_atoms,)
    int device_index,
    const char* model_type,
    int network_index,
    bool use_double_precision,
    bool use_cuda_device,
    bool use_cuaev
) {
    if (use_cuaev and not use_cuda_device) {
        std::cerr << "WARNING (libtorchani)\n"
                  << "A CUDA capable device must be selected to use the cuAEV\n"
                  << "but CPU was selected.\n"
                  << "The cuAEV extension will be disabled"
                  << std::endl;
        use_cuaev = false;
    }
    std::string jit_models_dir = __FILE__;
    // The following is roughtly equivalent to dir.parent.parent
    jit_models_dir = jit_models_dir.substr(0, jit_models_dir.find_last_of("/"));
    jit_models_dir = jit_models_dir.substr(0, jit_models_dir.find_last_of("/"));
    jit_models_dir = jit_models_dir + "/jit";

    std::string model_type_str = model_type;
    // Path to extra features may be inside model_type_str, separated by ::
    std::size_t loc = model_type_str.find("::");
    std::string extra_features_fpath = "";
    if (loc != std::string::npos) {
        extra_features_fpath = model_type_str.substr(loc + 2);
        model_type_str = model_type_str.substr(0, loc);
        // Use internal test file in this case
        if (extra_features_fpath == "ace-ala-nme-test") {
            extra_features_fpath = jit_models_dir + "/ace-ala-nme.charges";
        }
    }

    std::string model_path = "";
    if (std::find(
            torchani_builtin_models.begin(), torchani_builtin_models.end(), model_type_str
        ) != torchani_builtin_models.end()) {
        std::string jit_model_fname = model_type_str + ".pt";
        //  AimNet2 models are stored in a subdirectory
        std::string aimnet2_prefix = "aimnet2";
        std::string nutmeg_prefix = "nutmeg";
        if (jit_model_fname.compare(0, aimnet2_prefix.size(), aimnet2_prefix) == 0) {
            model_path = jit_models_dir + "/aimnet2" + "/" + jit_model_fname;
        } else if (jit_model_fname.compare(0, nutmeg_prefix.size(), nutmeg_prefix) ==
                   0) {
            model_path = jit_models_dir + "/nutmeg" + "/" + jit_model_fname;
        } else {
            model_path = jit_models_dir + "/" + jit_model_fname;
        }
        LOG(INFO) << "MLIP Model file name: " << jit_model_fname;
    } else {
        // Assume the path has been passed directly
        model_path = model_type_str;
    }
    global_config.setDeviceAndPrecision(
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
    torchani_atomic_numbers = torch::from_blob(
        atomic_nums, {num_atoms}, torch::TensorOptions().dtype(torch::kInt)
    );

    auto no_eval_atoms_mask = torch::from_blob(
        no_eval_atoms, {num_atoms}, torch::TensorOptions().dtype(torch::kInt)
    ).to(torch::kBool).to(global_config.device());
    bool use_no_eval_atoms = no_eval_atoms_mask.any().item().toBool();
    global_config.setNoEvalAtoms(use_no_eval_atoms);
    torchani_do_eval_atoms_mask = ~no_eval_atoms_mask;
    torchani_do_eval_idxs = torchani_do_eval_atoms_mask.nonzero().view({-1});
    LOG(INFO) << "MLIP Model full path: " << model_path;
    torchani_atomic_numbers = torchani_atomic_numbers.to(global_config.device());
    torchani_atomic_numbers = torchani_atomic_numbers.to(torch::kLong);
    // This is necessary since torch has to use this tensor to index internally,
    // and only long tensors can be used for this, so widening of the ints has
    // to be performed.
    // Also, torchani needs an extra dimension as batch dimension
    torchani_atomic_numbers = torchani_atomic_numbers.unsqueeze(0);
    LOG(INFO) << "Initialized MLIP with atomic numbers:\n" << torchani_atomic_numbers;
    // Discard unused idxs
    torchani_atomic_numbers = torchani_atomic_numbers.index_select(1, {torchani_do_eval_idxs});
    LOG(INFO) << "After discarding unused atoms, atomic numbers are:\n" << torchani_atomic_numbers;

    // Disable TF32 and FP16 for accuracy
    disableTorchReducedFloatPrecisionOptimizations();

    // The model is loaded from a pytorch file.
    try {
        model = mlip::MlipModel::from_model_path(model_path, global_config.device(), global_config.dtype());
    } catch (const std::exception& e) {
        std::cerr
            << "ERROR (libtorchani)\n"
            << "Could not load model correctly from path: " << model_path << "\n"
            << "Make sure the path is absolute, and correctly specified,\n"
            << "or that you are correctly specifying one of the built-in models.\n"
            << "Also make sure the model has been correctly packaged\n"
            << std::endl;
        throw;
    }

    // Set the correct model configuration
    if (network_index != -1) {
        try {
            model->useOnlyEnsembleMember(network_index);
        } catch (const std::runtime_error& e) {
            std::cerr << "ERROR (libtorchani)\n"
                      << "You set 'network_index' to a value != -1, but:\n"
                      << e.what() << std::endl;
            std::exit(2);
        }
    }
    std::string strat = use_cuaev ? "cuaev" : "pyaev";
    try {
        model->setStrategy(strat);
    } catch (const std::runtime_error& e) {
        std::cerr << "WARNING (libtorchani)\n"
                  << e.what() << "\n"
                  << "The passed strategy was " << strat
                  << ". The cuAEV extension will be disabled.\n"
                  << " If you are not using a traditional ANI model with AEVs,"
                  << " you can safely disregard this warning."
                  << std::endl;
    }

    // Models can read arbitrary features from a newline separated
    // file with floats (e.g. Nutmeg potentials use this)
    if (not extra_features_fpath.empty()) {
        try {
            model->readExtraFeaturesFromFile(extra_features_fpath);
        } catch (const std::runtime_error& e) {
            std::cerr << "ERROR (libtorchani)\n"
                      << "You specified an extra features file '" << extra_features_fpath
                      << "', but:\n"
                      << e.what() << std::endl;
            std::exit(2);
        }
    }
    global_config.setTorchaniIsInit(true);
    LOG(INFO) << "Loaded MLIP model. Finished TorchANI-Amber MLIP initialization";
}

void torchani_calc_energy_force_from_external_neighbors(
    int num_atoms,
    int num_neighbors,
    double coords_buf[][3],
    int neighborlist_buf[],
    double shifts_buf[][3],
    int* molecule_idxs_buf,  // Shape (num_atoms,)
    bool calc_only_bonded,
    int net_charge,
    /* outputs */
    double forces_buf[][3],
    double* potential_energy_buf
) {
    exitIfNotInitialized(global_config, "torchani_calc_energy_force_from_external_neighbors");
    timing::ScopedTimer timer{"energy force external neighbors"};
    if (!model->supportsComputeFromExternalNeighbors()) {
        std::cerr << "ERROR (libtorchani)\n"
                  << "To use an external neighborlist (use_amber_neighborlist=.true.)\n"
                  << " the model must export 'compute_from_external_neighbors(...)'."
                  << " Consult the TorchANI-Amber README.md for the required signature\n"
                  << "https://github.com/roitberg-group/torchani-amber" << std::endl;
        std::exit(2);
    }
    torch::Tensor coords = tensor_utils::dblBufToCoordsTensor(global_config, coords_buf, num_atoms);
    torch::Tensor shifts = tensor_utils::dblBufToShiftsTensor(global_config, shifts_buf, num_neighbors);
    torch::Tensor neighborlist =
        tensor_utils::intBufToNeighborlistTensor(global_config, neighborlist_buf, num_neighbors);
    torch::Tensor molecule_idxs =
        tensor_utils::intBufToI64Tensor(global_config, molecule_idxs_buf, {num_atoms});

    if (global_config.useNoEvalAtoms()) {
        auto mask = torchani_do_eval_atoms_mask.view({-1}).index({neighborlist.view(-1)}).view({2, -1});
        auto do_eval_pairs = (torch::all(mask, 0)).nonzero().flatten();
        neighborlist = neighborlist.index_select(1, {do_eval_pairs});
        shifts = shifts.index_select(0, {do_eval_pairs});
        torch::Tensor reduced_idxs = torch::full({num_atoms}, -1, neighborlist.options());
        reduced_idxs.index_put_(
            {torchani_do_eval_idxs},
            torch::arange(torchani_do_eval_idxs.size(0), neighborlist.options())
        );
        neighborlist = reduced_idxs.index({neighborlist});
        molecule_idxs = molecule_idxs.index({torchani_do_eval_idxs});
        coords = coords.index_select(1, {torchani_do_eval_idxs});
    }

    std::optional<torch::Tensor> maybe_molecule_idxs;
    if (calc_only_bonded or global_config.useNoEvalAtoms()) {
        maybe_molecule_idxs = molecule_idxs;
    }
    mlip::MlipOutput output = model->computeFromExternalNeighbors(
        torchani_atomic_numbers,
        coords,
        neighborlist,
        shifts,
        net_charge,
        /*atomic=*/false,
        /*ensemble_values=*/false,
        maybe_molecule_idxs,
        /*calc_forces=*/true
    );
    torch::Tensor energy = output.at("energy");
    populateForces(output.at("forces"), forces_buf, num_atoms);
    populateEnergyScalar(energy, potential_energy_buf);
}

void torchani_calc_energy_force(
    int num_atoms,
    double coords_buf[][3],
    double cell_buf[][3],
    bool use_pbc,
    int* molecule_idxs_buf,
    bool calc_only_bonded,
    int net_charge,
    double forces_buf[][3],
    double* potential_energy
) {
    exitIfNotInitialized(global_config, "torchani_calc_energy_force");
    timing::ScopedTimer timer{"energy force"};
    torch::Tensor coords = tensor_utils::dblBufToCoordsTensor(global_config, coords_buf, num_atoms);
    torch::Tensor cell = tensor_utils::dblBufToCellTensor(global_config, cell_buf);
    torch::Tensor molecule_idxs =
        tensor_utils::intBufToI64Tensor(global_config, molecule_idxs_buf, {num_atoms});
    if (global_config.useNoEvalAtoms()) {
        molecule_idxs = molecule_idxs.index({torchani_do_eval_idxs});
        coords = coords.index_select(1, {torchani_do_eval_idxs});
    }
    // Inputs are setup with PBC
    std::optional<torch::Tensor> maybe_cell;
    if (use_pbc) {
        maybe_cell = cell;
    }
    // In the no_eval_atoms case, the energies will still be added, but the
    // atoms will feel no forces
    std::optional<torch::Tensor> maybe_molecule_idxs;
    if (calc_only_bonded or global_config.useNoEvalAtoms()) {
        maybe_molecule_idxs = molecule_idxs;
    }
    mlip::MlipOutput output = model->computeSinglePoint(
        torchani_atomic_numbers,
        coords,
        maybe_cell,
        net_charge,
        /*atomic=*/false,
        /*ensemble_values=*/false,
        maybe_molecule_idxs,
        /*calc_forces=*/true
    );
    torch::Tensor energy = output.at("energy");
    populateForces(output.at("forces"), forces_buf, num_atoms);
    populateEnergyScalar(energy, potential_energy);
}

void torchani_energy_force_atomic_charges(
    int num_atoms,
    double coords_buf[][3],
    /* outputs */
    double forces_buf[][3],
    double atomic_charges_buf[],
    double* potential_energy
) {
    exitIfNotInitialized(global_config, "torchani_energy_force_atomic_charges");
    exitIfNoEvalAtoms(global_config, "torchani_energy_force_atomic_charges");
    torch::Tensor coords = tensor_utils::dblBufToCoordsTensor(global_config, coords_buf, num_atoms);
    mlip::MlipOutput output = model->computeSinglePoint(
        torchani_atomic_numbers,
        coords,
        /*cell=*/std::nullopt,
        /*net_charge=*/0,
        /*atomic=*/false,
        /*ensemble_values=*/false,
        /*molecule_idxs=*/std::nullopt,
        /*calc_forces=*/true,
        /*calc_charges=*/true
    );
    torch::Tensor energy = output.at("energy");
    torch::Tensor atomic_charges = output.at("atomic_charges");

    populateForces(output.at("forces"), forces_buf, num_atoms);
    populateEnergyScalar(energy, potential_energy);
    populateAtomicCharges(atomic_charges, atomic_charges_buf, num_atoms);
}

/**
 * Note that currently this function can't split the QM forces in two parts, it only
 * outputs net forces due to the model in vacuum and the potential energies
 *
 * Currently this is the only code path that supports charged molecules
 */
void torchani_calc_energy_force_with_coupling(
    int num_atoms,
    int num_env_charges,
    double inv_pol_dielectric,
    double coords_buf[][3],
    double atomic_alphas_buf[],  // shape (num-atoms,)
    double env_charge_coords_buf[][3],  //  shape (num-charges, 3)
    double env_charges_buf[],  // shape (num-charges,)
    bool predict_charges,
    bool simple_polarization_correction,
    bool use_charge_derivatives,
    int ml_system_charge,
    /* outputs */
    double forces_on_atoms_buf[][3],  // shape (num-atoms, 3)
    double forces_on_env_charges_buf[][3],  // shape (num-charges, 3)
    double atomic_charges_buf[],  // shape (num-atoms,) (actually in-out)
    double* ene_pot_invacuo_buf,
    double* ene_pot_embed_pol_buf,
    double* ene_pot_embed_dist_buf,
    double* ene_pot_embed_coulomb_buf,
    double* ene_pot_total_buf
) {
    exitIfNotInitialized(global_config, "torchani_calc_energy_force_with_coupling");
    exitIfNoEvalAtoms(global_config, "torchani_calc_energy_force_with_coupling");
    torch::Tensor coords = tensor_utils::dblBufToCoordsTensor(global_config, coords_buf, num_atoms);
    torch::Tensor atomic_charges = tensor_utils::dblBufToTensor(global_config, atomic_charges_buf, {num_atoms});

    mlip::MlipOutput output = model->computeSinglePoint(
        torchani_atomic_numbers,
        coords,
        /*cell=*/std::nullopt,
        ml_system_charge,
        /*atomic=*/false,
        /*ensemble_values=*/false,
        /*molecule_idxs=*/std::nullopt,
        /*calc_forces=*/false,
        /*calc_charges=*/predict_charges
    );
    torch::Tensor ene_pot_invacuo = output.at("energy");

    if (predict_charges) {
        atomic_charges = output.at("atomic_charges");
        if (not use_charge_derivatives) {
            // Disregard dependence of predicted charges on coordinates
            atomic_charges.detach_();
        }
    }

    // Embedding part
    torch::Tensor atomic_alphas =
        tensor_utils::dblBufToTensor(global_config, atomic_alphas_buf, {num_atoms});
    torch::Tensor env_charge_coords =
        tensor_utils::dblBufToCoordsTensor(global_config, env_charge_coords_buf, num_env_charges);
    torch::Tensor env_charges =
        tensor_utils::dblBufToTensor(global_config, env_charges_buf, {num_env_charges});

    torch::Tensor delta = torch::reshape(env_charge_coords, {-1, 1, 3}) -
        torch::reshape(coords, {1, -1, 3});
    torch::Tensor env_charges_to_atoms_distances =
        torch::linalg_norm(delta, std::nullopt, 2);

    // Embedding calculations are made in atomic units, so outputs are in Ha
    auto ene_pot_coulomb = electro::coulombicEmbeddingEnergy(
        atomic_charges, env_charges, env_charges_to_atoms_distances
    );
    torch::Tensor ene_pot_pol =
        torch::zeros(1, torch::dtype(global_config.dtype()).device(global_config.device()));
    torch::Tensor ene_pot_dist =
        torch::zeros(1, torch::dtype(global_config.dtype()).device(global_config.device()));
    if (simple_polarization_correction) {
        ene_pot_pol = electro::polarizableEmbeddingEnergy(
            coords,
            atomic_alphas,
            env_charge_coords,
            env_charges,
            env_charges_to_atoms_distances,
            inv_pol_dielectric
        );
        ene_pot_dist = -0.5 * ene_pot_pol;
    }

    auto ene_pot_total = ene_pot_invacuo + ene_pot_dist + ene_pot_pol + ene_pot_coulomb;
    // TODO It may be possible to return forces due to "different things"
    // by detaching the coords before the final calculation

    // TODO: Retain grad in atomic_charges so that the derivatives
    // can be returned. Use backward instead of autograd::grad for simplicity to do
    // this maybe?
    std::vector<torch::Tensor> energy_vec = {ene_pot_total.sum()};
    std::vector<torch::Tensor> coord_vec = {coords, env_charge_coords};
    // the output is a vector of tensors, one per element of coord_vec, so the
    // [0] and [1] elements are the derivatives wrt the atom and env charge
    // coords, and their negatives are the forces
    auto ad_output = torch::autograd::grad(
        energy_vec,
        coord_vec,
        /* grad_outputs= */ {},
        /* retain_graph= */ false
    );
    populateEmbeddingForces(
        -ad_output[0],
        -ad_output[1],
        num_atoms,
        num_env_charges,
        forces_on_atoms_buf,
        forces_on_env_charges_buf
    );
    populateEnergyScalar(ene_pot_total, ene_pot_total_buf);
    populateEnergyScalar(ene_pot_invacuo, ene_pot_invacuo_buf);
    populateEnergyScalar(ene_pot_pol, ene_pot_embed_pol_buf);
    populateEnergyScalar(ene_pot_dist, ene_pot_embed_dist_buf);
    populateEnergyScalar(ene_pot_coulomb, ene_pot_embed_coulomb_buf);
    if (predict_charges) {
        populateAtomicCharges(atomic_charges, atomic_charges_buf, num_atoms);
    }
}


void torchani_calc_energy_force_custom_mlmm(
    int num_atoms,
    int num_env_charges,
    double coords_buf[][3],
    double env_charge_coords_buf[][3],
    double env_charges_buf[],
    double cell_buf[][3],
    bool use_pbc,
    int* molecule_idxs_buf,
    bool calc_only_bonded,
    int ml_system_charge,
    /* outputs */
    double forces_on_atoms_buf[][3],
    double forces_on_env_charges_buf[][3],
    double* ene_pot_total_buf
) {
    exitIfNotInitialized(global_config, "torchani_calc_energy_force_custom_mlmm");
    exitIfNoEvalAtoms(global_config, "torchani_calc_energy_force_custom_mlmm");
    if (!model->supportsCustomMlmm()) {
        std::cerr << "ERROR (libtorchani)\n"
                  << "mlmm_coupling=2 requires the model to export 'compute_mlmm(...)'. "
                  << " Consult the TorchANI-Amber README.md for the required signature\n"
                  << "https://github.com/roitberg-group/torchani-amber" << std::endl;
        std::exit(2);
    }
    torch::Tensor ml_coords = tensor_utils::dblBufToCoordsTensor(global_config, coords_buf, num_atoms);
    torch::Tensor mm_coords = tensor_utils::dblBufToCoordsTensor(global_config, env_charge_coords_buf, num_env_charges);
    torch::Tensor mm_charges = tensor_utils::dblBufToTensor(global_config, env_charges_buf, {1, num_env_charges});
    torch::Tensor cell = tensor_utils::dblBufToCellTensor(global_config, cell_buf);
    torch::Tensor molecule_idxs =
        tensor_utils::intBufToI64Tensor(global_config, molecule_idxs_buf, {num_atoms});

    std::optional<torch::Tensor> maybe_cell;
    if (use_pbc) {
        maybe_cell = cell;
    }
    std::optional<torch::Tensor> maybe_molecule_idxs;
    if (calc_only_bonded) {
        maybe_molecule_idxs = molecule_idxs;
    }
    mlip::MlipOutput output = model->computeMlmm(
        torchani_atomic_numbers,
        ml_coords,
        mm_coords,
        mm_charges,
        maybe_cell,
        ml_system_charge,
        /*atomic=*/false,
        /*ensemble_values=*/false,
        maybe_molecule_idxs,
        /*calc_forces=*/true
    );
    torch::Tensor energy = output.at("energy");

    populateEmbeddingForces(
        output.at("forces"), output.at("mm_forces"), num_atoms, num_env_charges,
        forces_on_atoms_buf, forces_on_env_charges_buf
    );
    populateEnergyScalar(energy, ene_pot_total_buf);
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
    exitIfNotInitialized(global_config, "torchani_energy_force_atomic_charges_with_derivatives");
    exitIfNoEvalAtoms(global_config, "torchani_energy_force_atomic_charges_with_derivatives");
    torch::Tensor coords = tensor_utils::dblBufToCoordsTensor(global_config, coords_buf, num_atoms);
    mlip::MlipOutput output = model->computeSinglePoint(
        torchani_atomic_numbers,
        coords,
        /*cell=*/std::nullopt,
        /*net_charge=*/0,
        /*atomic=*/false,
        /*ensemble_values=*/false,
        /*molecule_idxs=*/std::nullopt,
        /*calc_forces=*/true,
        /*calc_charges=*/true,
        /*calc_charges_grad=*/true
    );
    torch::Tensor energy = output.at("energy");
    torch::Tensor atomic_charges = output.at("atomic_charges");

    populateForces(output.at("forces"), forces_buf, num_atoms);
    populateAtomicChargeDerivatives(
        output.at("atomic_charge_grads"), atomic_charge_derivatives, num_atoms
    );
    populateEnergyScalar(energy, potential_energy);
    populateAtomicCharges(atomic_charges, atomic_charges_buf, num_atoms);
}

void torchani_energy_atomic_charges_retain(
    int num_atoms,
    double coords_buf[][3],
    int net_charge,
    /* outputs */
    double atomic_charges_buf[],
    double* potential_energy
) {
    exitIfNotInitialized(global_config, "torchani_energy_atomic_charges_retain");
    exitIfNoEvalAtoms(global_config, "torchani_energy_atomic_charges_retain");
    // The graph outlives this call, so it must not refer to the caller's buffer
    // (dblBufToCoordsTensor does not copy for CPU double precision)
    torch::Tensor coords = tensor_utils::dblBufToCoordsTensor(global_config, coords_buf, num_atoms)
                               .detach()
                               .clone()
                               .requires_grad_(true);
    mlip::MlipOutput output = model->computeSinglePoint(
        torchani_atomic_numbers,
        coords,
        /*cell=*/std::nullopt,
        net_charge,
        /*atomic=*/false,
        /*ensemble_values=*/false,
        /*molecule_idxs=*/std::nullopt,
        /*calc_forces=*/false,
        /*calc_charges=*/true
    );
    torch::Tensor energy = output.at("energy");
    torch::Tensor atomic_charges = output.at("atomic_charges");

    populateEnergyScalar(energy, potential_energy);
    // populateAtomicCharges converts its argument to CPU double, keep the
    // original tensor (and its graph) for the backward pass
    torch::Tensor charges_copy = atomic_charges;
    populateAtomicCharges(charges_copy, atomic_charges_buf, num_atoms);
    retained_coords = coords;
    retained_energy = energy;
    retained_atomic_charges = atomic_charges;
    charges_graph_retained = true;
}

void torchani_energy_atomic_charges_vjp(
    int num_atoms,
    double charge_weights_buf[],
    /* outputs */
    double forces_buf[][3]
) {
    exitIfNotInitialized(global_config, "torchani_energy_atomic_charges_vjp");
    if (not charges_graph_retained) {
        std::cerr << "ERROR (libtorchani)\n"
                  << "torchani_energy_atomic_charges_vjp must follow a call to"
                  << " torchani_energy_atomic_charges_retain" << std::endl;
        std::exit(2);
    }
    torch::Tensor weights =
        tensor_utils::dblBufToTensor(global_config, charge_weights_buf, {1, num_atoms});
    std::vector<torch::Tensor> output_vec = {
        retained_energy.sum() * HARTREE_TO_KCALMOL +
        (retained_atomic_charges * weights).sum()
    };
    std::vector<torch::Tensor> coord_vec = {retained_coords};
    torch::Tensor forces = -torch::autograd::grad(
                               output_vec,
                               coord_vec,
                               /* grad_outputs= */ {},
                               /* retain_graph= */ false
                           )[0];
    forces = forces.to(torch::kCPU, torch::kDouble);
    auto forces_a = forces.accessor<double, 3>();
    for (int atom = 0; atom != num_atoms; ++atom) {
        for (int c = 0; c != 3; ++c) {
            forces_buf[atom][c] = forces_a[0][atom][c];
        }
    }
    retained_coords = torch::Tensor();
    retained_energy = torch::Tensor();
    retained_atomic_charges = torch::Tensor();
    charges_graph_retained = false;
}

void torchani_calc_energy_force_qbc(
    int num_atoms,
    double coords_buf[][3],
    int net_charge, 
    double forces_buf[][3],
    double* qbc_buf,
    double qbc_derivatives[][3],
    double* potential_energy
) {
    exitIfNotInitialized(global_config, "torchani_calc_energy_force_qbc");
    exitIfNoEvalAtoms(global_config, "torchani_calc_energy_force_qbc");
    torch::Tensor coords = tensor_utils::dblBufToCoordsTensor(global_config, coords_buf, num_atoms);
    mlip::MlipOutput output = model->computeSinglePoint(
        torchani_atomic_numbers,
        coords,
        /*cell=*/std::nullopt,
        net_charge,
        /*atomic=*/false,
        /*ensemble_values=*/false,
        /*molecule_idxs=*/std::nullopt,
        /*calc_forces=*/true,
        /*calc_charges=*/false,
        /*calc_charges_grad=*/false,
        /*calc_qbc_and_qbc_grad=*/true
    );
    torch::Tensor energy = output.at("energy");
    torch::Tensor qbc = output.at("qbc");
    populateQbcDerivatives(output.at("qbc_grads"), qbc_derivatives, num_atoms);
    populateForces(output.at("forces"), forces_buf, num_atoms);
    populateEnergyScalar(energy, potential_energy);
    populateEnergyScalar(qbc, qbc_buf);
}

void torchani_calc_data_for_monitored_mlmm(
    int num_atoms,
    double coords_buf[][3],
    int net_charge,
    /* outputs */
    double forces_buf[][3],
    double atomic_charges_buf[],
    /* TODO not sure how to specify type of atomic_charges_grad, besides "buf" */
    double* atomic_charge_derivatives,
    double* qbc_buf,
    double qbc_derivatives[][3],
    double* potential_energy
) {
    exitIfNotInitialized(global_config, "torchani_calc_data_for_monitored_mlmm");
    exitIfNoEvalAtoms(global_config, "torchani_calc_data_for_monitored_mlmm");
    torch::Tensor coords = tensor_utils::dblBufToCoordsTensor(global_config, coords_buf, num_atoms);
    mlip::MlipOutput output = model->computeSinglePoint(
        torchani_atomic_numbers,
        coords,
        /*cell=*/std::nullopt,
        net_charge,
        /*atomic=*/false,
        /*ensemble_values=*/false,
        /*molecule_idxs=*/std::nullopt,
        /*calc_forces=*/true,
        /*calc_charges=*/true,
        /*calc_charges_grad=*/true,
        /*calc_qbc_and_qbc_grad=*/true
    );
    torch::Tensor energy = output.at("energy");
    torch::Tensor atomic_charges = output.at("atomic_charges");
    torch::Tensor qbc = output.at("qbc");

    populateForces(output.at("forces"), forces_buf, num_atoms);
    populateQbcDerivatives(output.at("qbc_grads"), qbc_derivatives, num_atoms);
    populateEnergyScalar(energy, potential_energy);
    populateEnergyScalar(qbc, qbc_buf);
    populateAtomicChargeDerivatives(
        output.at("atomic_charge_grads"), atomic_charge_derivatives, num_atoms
    );
    populateAtomicCharges(atomic_charges, atomic_charges_buf, num_atoms);
}
}
