#pragma once

extern "C" {
void torchani_initialize(
    int num_atoms,
    int atomic_nums[],  // Shape (num_atoms,)
    int no_eval_atoms[],  // Shape (num_atoms,)
    const char* model_type,
    int device_index,
    int network_index,
    bool use_double_precision,
    bool use_cuda_device,
    bool use_cuaev
);

void torchani_calc_energy_force(
    int num_atoms,
    double coords[][3],  // Shape (num_atoms, 3)
    double cell[][3],  // Shape (3, 3)
    bool use_pbc,
    int* molecule_idxs_buf, // Shape (num_atoms,)
    bool calc_only_bonded,
    int net_charge,
    /* outputs */
    double forces[][3],  // Shape (num_atoms, 3)
    double* potential_energy  // Scalar
);

// General full-ML scalar-output entry point for new callers. This covers the
// charge-only use cases handled by torchani_energy_force_atomic_charges* while
// also supporting volumes, PBC/cell inputs, molecule indices, and net charge.
// The older charge-specific functions remain available for compatibility.
//
// atomic_*_grad arrays are [num_atoms x num_atoms x 3].
// In Fortran, pass arrays of shape [3, num_atoms, num_atoms], where element
// [i, j, k] is the derivative of the scalar on atom k with respect to
// coordinate i of atom j.
void torchani_calc_energy_force_atomic_scalars(
    int num_atoms,
    double coords[][3],  // Shape (num_atoms, 3)
    double cell[][3],  // Shape (3, 3)
    bool use_pbc,
    int* molecule_idxs_buf, // Shape (num_atoms,)
    bool calc_only_bonded,
    int net_charge,
    bool write_charges,
    bool write_charges_grad,
    bool write_volumes,
    bool write_volumes_grad,
    /* outputs */
    double forces[][3],  // Shape (num_atoms, 3)
    double atomic_charges[],  // Shape (num_atoms,)
    double* atomic_charges_grad,
    double atomic_volumes[],  // Shape (num_atoms,)
    double* atomic_volumes_grad,
    double* potential_energy  // Scalar
);

void torchani_calc_energy_force_from_external_neighbors(
    int num_atoms,
    int num_neighbors,
    double coords[][3],  // Shape (num_atoms, 3)
    int* neighborlist[2],  // Shape (2, num_atoms)
    double shifts[][3], // Shape (num_atoms, 3)
    int* molecule_idxs_buf, // Shape (num_atoms,)
    bool calc_only_bonded,
    int net_charge,
    /* outputs */
    double forces[][3],  // Shape (num_atoms, 3)
    double* potential_energy
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

// Backwards compat, dummy fn, remove when AmberTools26 is released
void torchani_energy_force_qbc(
    int num_atoms,
    double coords[][3],  // Shape (num_atoms, 3)
    /* outputs */
    double forces[][3],  // Shape (num_atoms, 3)
    double* potential_energy,  // Scalar
    double* qbc, // Scalar
    double qbc_grad[][3]  // Shape (num_atoms, 3)
){};

// Backwards compat, dummy fn, remove when AmberTools26 is released
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
) {};


void torchani_calc_energy_force_qbc(
    int num_atoms,
    double coords[][3],  // Shape (num_atoms, 3)
    int net_charge,
    /* outputs */
    double forces[][3],  // Shape (num_atoms, 3)
    double* qbc, // Scalar
    double qbc_grad[][3],  // Shape (num_atoms, 3)
    double* potential_energy  // Scalar
);

void torchani_calc_data_for_monitored_mlmm(
    int num_atoms,
    double coords[][3],  // Shape (num_atoms, 3)
    int net_charge,
    /* outputs */
    double forces[][3],  // Shape (num_atoms, 3)
    double atomic_charges[],  // Shape (num_atoms,)
    double* atomic_charges_grad,  // Shape (num_atoms, num_atoms, 3) TODO: Type properly
    double* qbc,  // Scalar
    double qbc_grad[][3],  // Shape (num_atoms, 3)
    double* potential_energy  // Scalar
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

// efield_buf has shape [num_atoms, 3] and stores the MM electric field at
// each ML atom in atomic units. efield_grad_mm_buf is compatible with a
// Fortran array of shape [3, num_env_charges, 3, num_atoms], where element
// [beta, j, alpha, i] is d E_alpha(i) / d R_beta(j), in a.u./Angstrom.
void torchani_calc_energy_force_with_coupling(
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
    bool predict_volumes,
    int ml_system_charge,
    bool write_charges,
    bool write_charges_grad,
    bool write_volumes,
    bool write_volumes_grad,
    bool write_efield,
    bool write_efield_grad_mm,
    /* outputs */
    double forces_on_atoms_buf[][3],  // shape (num-atoms, 3)
    double forces_on_env_charges_buf[][3],  // shape (num-charges, 3)
    double atomic_charges_buf[],  // shape (num-atoms, 3)
    double* atomic_charges_grad_buf,
    double atomic_volumes_buf[],  // shape (num-atoms,)
    double* atomic_volumes_grad_buf,
    double efield_buf[][3],
    double* efield_grad_mm_buf,
    double* ene_pot_invacuo_buf,
    double* ene_pot_embed_pol_buf,
    double* ene_pot_embed_dist_buf,
    double* ene_pot_embed_coulomb_buf,
    double* ene_pot_total_buf
);
}
