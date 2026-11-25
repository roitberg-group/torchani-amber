#include "electro.h"

#include <cmath>

#include <torch/all.h>

#include "constants.h"

namespace {
// TODO: dividing by ...distances probably needs an epsilon
/**
 * Calculate the electric field that the MM atom charges, located at 'mm_coords',
 * generate on the ML atoms, located at 'ml_coords'. Shape of 'ml_coords' and
 * 'mm_coords' doesn't need to match, but first dim of 'mm_coords' and
 * 'mm_atomic_charges' must match
 * Output has shape (ml_atoms_num, 3)
 *
 * ml_coords: shape (ml_atoms_num, 3)
 * mm_coords: shape (mm_atoms_num, 3)
 * mm_atomic_charges: shape (mm_atoms_num,)
 * mm_to_ml_distances: shape (mm_atoms_num, ml_atoms_num)
 */
auto calcEfield(
    torch::Tensor ml_coords,
    torch::Tensor mm_coords,
    torch::Tensor mm_atomic_charges,
    torch::Tensor mm_to_ml_distances
) -> torch::Tensor {
    auto delta_coords = torch::reshape(mm_coords, {-1, 1, 3}) -
        torch::reshape(ml_coords, {1, -1, 3});
    auto efield = torch::reshape(mm_atomic_charges, {-1, 1, 1}) * delta_coords /
        mm_to_ml_distances.unsqueeze(-1).pow(3);
    // Inputs are Ang for *coords, and e for mm_atomic_charges
    // Electric fields with angstrom units are unintuitive, to avoid that
    // Output in AU (Hartree / e), so perform the conversion:
    // E_field_AU = E_field_Angstrom / A2B**2
    return torch::sum(efield, 0) / std::pow(constants::CODATA08_ANGSTROM_TO_BOHR, 2);
}

/**
 * Given efield and alphas, calculate the associated QM/MM energy
 */
auto polarizableEmbeddingEnergyFromField(
    torch::Tensor atomic_alphas, torch::Tensor efield, double inv_pol_dielectric
) -> torch::Tensor {
    // Input atomic alphas and Efield are in AU
    return -inv_pol_dielectric *
        torch::sum(atomic_alphas * torch::sum(efield.pow(2), 1));
}
}  // namespace

/**
 * API
 */
namespace electro {
auto polarizableEmbeddingEnergy(
    torch::Tensor ml_coords,
    torch::Tensor atomic_alphas,
    torch::Tensor mm_coords,
    torch::Tensor mm_atomic_charges,
    torch::Tensor mm_to_ml_distances,
    double inv_pol_dielectric
) -> torch::Tensor {
    torch::Tensor efield = calcEfield(
        ml_coords, mm_coords, mm_atomic_charges, mm_to_ml_distances
    );
    // Output in AU (Hartree), no conversion needed, since inputs are fw to a
    // fn that outputs AU
    return polarizableEmbeddingEnergyFromField(
        atomic_alphas, efield, inv_pol_dielectric
    );
}
auto coulombicEmbeddingEnergy(
    torch::Tensor ml_atomic_charges,
    torch::Tensor mm_atomic_charges,
    torch::Tensor mm_to_ml_distances
) -> torch::Tensor {
    torch::Tensor pair_energies = torch::reshape(mm_atomic_charges, {-1, 1}) *
        torch::reshape(ml_atomic_charges, {1, -1}) / mm_to_ml_distances;
    // Output in AU (Hartree), so perform the conversion
    // Epot_AU = E_pot_Angstrom / A2B
    return torch::sum(pair_energies) / constants::CODATA08_ANGSTROM_TO_BOHR;
}
}  // namespace electro
