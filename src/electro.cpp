#include "electro.h"

#include <cmath>

#include <torch/all.h>
#include <torch/script.h>

namespace {
// Factor used by Amber
constexpr double CODATA08_ANGSTROM_TO_BOHR = 1.8897261328727875;
// TODO: dividing by ...distances probably needs an epsilon
/**
 * Calculate the electric field that a sequence of charges in an 'environment' (env),
 * located at 'env_charge_coords', generates on some other 'coords'. Shape of 'coords'
 * and 'env_charge_coords' doesn't need to match, but first dim of 'env_charge_coords'
 * and 'env_charges' must match
 * Output has shape (num-atoms, 3)
 *
 * coords: shape (num-atoms, 3)
 * env_charge_coords: shape (num-charges, 3)
 * env_charges: shape (num-charges,)
 * env_charges_to_atoms_distances: shape (num-charges, num-atoms)
 */
auto calc_efield(
    torch::Tensor coords,
    torch::Tensor env_charge_coords,
    torch::Tensor env_charges,
    torch::Tensor env_charges_to_atoms_distances
) -> torch::Tensor {
    auto delta_coords = torch::reshape(env_charge_coords, {-1, 1, 3}) -
        torch::reshape(coords, {1, -1, 3});
    auto efield = torch::reshape(env_charges, {-1, 1, 1}) * delta_coords /
        env_charges_to_atoms_distances.unsqueeze(-1).pow(3);
    // Inputs are Ang for *coords, and e for env_charges
    // Electric fields with angstrom units are unintuitive, to avoid that
    // Output in AU (Hartree / e), so perform the conversion:
    // E_field_AU = E_field_Angstrom / A2B**2
    return torch::sum(efield, 0) / std::pow(CODATA08_ANGSTROM_TO_BOHR, 2);
}

/**
 * Given efield and alphas, calculate the associated QM/MM energy
 */
auto polarizable_embedding_energy_from_field(
    torch::Tensor atomic_alphas, torch::Tensor efield, double inv_pol_dielectric
) -> torch::Tensor {
    // Input atomic alphas are in Ang and Efield in AU
    // Output in AU, so perform the conversion:
    // Alphas_AU = Alphas_Angstrom * A2B**3
    torch::Tensor atomic_alphas_bohr =
        atomic_alphas * std::pow(CODATA08_ANGSTROM_TO_BOHR, 3);
    return -inv_pol_dielectric *
        torch::sum(atomic_alphas_bohr * torch::sum(efield.pow(2), 1));
}
}  // namespace

/**
 * API
 */
namespace electro {
auto polarizable_embedding_energy(
    torch::Tensor coords,
    torch::Tensor atomic_alphas,
    torch::Tensor env_charge_coords,
    torch::Tensor env_charges,
    torch::Tensor env_charges_to_atoms_distances,
    double inv_pol_dielectric
) -> torch::Tensor {
    torch::Tensor efield = calc_efield(
        coords, env_charge_coords, env_charges, env_charges_to_atoms_distances
    );
    // Output in AU (Hartree), no conversion needed, since inputs are fw to a
    // fn that outputs AU
    return polarizable_embedding_energy_from_field(
        atomic_alphas, efield, inv_pol_dielectric
    );
}
auto coulombic_embedding_energy(
    torch::Tensor atomic_charges,
    torch::Tensor env_charges,
    torch::Tensor env_charges_to_atoms_distances
) -> torch::Tensor {
    torch::Tensor pair_energies = torch::reshape(env_charges, {-1, 1}) *
        torch::reshape(atomic_charges, {1, -1}) / env_charges_to_atoms_distances;
    // Output in AU (Hartree), so perform the conversion
    // Epot_AU = E_pot_Angstrom / A2B
    return torch::sum(pair_energies) / CODATA08_ANGSTROM_TO_BOHR;
}
}  // namespace electro
