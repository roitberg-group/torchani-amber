#pragma once

extern "C" {
void torchani_init_model(
    int num_atoms,
    int atomic_nums[],  // Shape (num_atoms,)
    const char* model_type,
    int device_index,
    int network_index,
    bool use_double_precision,
    bool use_cuda_device,
    bool use_cuaev
);

void torchani_energy_force_atomic_charges(
    int num_atoms,
    double coords[][3],  // Shape (num_atoms, 3)
    /* outputs */
    double forces[][3],  // Shape (num_atoms, 3)
    double atomic_charges[],  // Shape (num_atoms,)
    double* potential_energy  // Scalar
);

// Note that atomic_charges_grad is a [num_atoms x num_atoms x 3]
// array In fortran, an array of shape [3, num_atoms, num_atoms] should be
// passed Where the element [i, j, k] is the derivative of the **charge on
// k-th atom** with respect to the **i-th position of the j-th atom**
void torchani_energy_force_atomic_charges_with_derivatives(
    int num_atoms,
    double coords[][3],  // Shape (num_atoms, 3)
    /* outputs */
    double forces[][3],  // Shape (num_atoms, 3)
    double atomic_charges[],  // Shape (num_atoms,)
    double* atomic_charges_grad,  // Shape (num_atoms, num_atoms, 3) TODO: Type properly
    double* potential_energy  // Scalar
);

void torchani_energy_force(
    int num_atoms,
    double coords[][3],  // Shape (num_atoms, 3)
    /* outputs */
    double forces[][3],  // Shape (num_atoms, 3)
    double* potential_energy  // Scalar
);

void torchani_energy_force_qbc(
    int num_atoms,
    double coords[][3],  // Shape (num_atoms, 3)
    /* outputs */
    double forces[][3],  // Shape (num_atoms, 3)
    double* qbc, // Scalar
    double* potential_energy  // Scalar
);

void torchani_energy_force_pbc(
    int num_atoms,
    double coords[][3],  // Shape (num_atoms, 3)
    double cell[][3],  // Shape (3, 3)
    /* outputs */
    double forces[][3],  // Shape (num_atoms, 3)
    double* potential_energy  // Scalar
);

void torchani_data_for_monitored_mlmm(
    int num_atoms,
    double coords[][3],  // Shape (num_atoms, 3)
    /* outputs */
    double forces[][3],  // Shape (num_atoms, 3)
    double atomic_charges[],  // Shape (num_atoms,)
    double* atomic_charges_grad,  // Shape (num_atoms, num_atoms, 3) TODO: Type properly
    double* qbc,  // Scalar
    double qbc_grad[][3],  // Shape (num_atoms, 3)
    double* potential_energy  // Scalar
);

void torchani_energy_force_from_external_neighbors(
    int num_atoms,
    int num_neighbors,
    double coords[][3],  // Shape (num_atoms, 3)
    int* neighborlist[2],  // Shape (2, num_atoms)
    double shifts[][3], // Shape (num_atoms, 3)
    /* outputs */
    double forces[][3],  // Shape (num_atoms, 3)
    double* potential_energy
);

void torchani_energy_force_with_coupling(
    int num_atoms,
    int num_env_charges,
    double distortion_k,
    double coords_buf[][3],
    double atomic_alphas_buf[],  // shape (num-atoms,)
    double env_charge_coords_buf[][3],  //  shape (num-charges, 3)
    double env_charges_buf[],  // shape (num-charges,)
    bool predict_charges,
    bool use_simple_polarization_correction,
    bool use_charge_derivatives,
    /* outputs */
    double forces_on_atoms_buf[][3],  // shape (num-atoms, 3)
    double forces_on_env_charges_buf[][3],  // shape (num-charges, 3)
    double atomic_charges_buf[],  // shape (num-atoms, 3)
    double* ene_pot_invacuo_buf,
    double* ene_pot_embed_pol_buf,
    double* ene_pot_embed_dist_buf,
    double* ene_pot_embed_coulomb_buf,
    double* ene_pot_total_buf
);
}
