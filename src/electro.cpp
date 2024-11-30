#include "electro.h"

#include <torch/all.h>
#include <torch/script.h>

namespace {
// TODO: There are a bunch of unnecessary calculations here, env_charge_coords and
// coords are not really needed to calculate the electric field energy
//
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
    return torch::sum(efield, 0);
}

/**
 * Given efield and alphas, calculate the associated QM/MM energy
 */
auto polarizable_embedding_energy_from_field(
    torch::Tensor atomic_alphas, torch::Tensor efield
) -> torch::Tensor {
    return -0.5 * torch::sum(atomic_alphas * torch::sum(efield.pow(2), 1));
}
}  // namespace anon

/**
 * API
 */
namespace electro {
auto polarizable_embedding_energy(
    torch::Tensor coords,
    torch::Tensor atomic_alphas,
    torch::Tensor env_charge_coords,
    torch::Tensor env_charges,
    torch::Tensor env_charges_to_atoms_distances
) -> torch::Tensor {
    torch::Tensor efield = calc_efield(
        coords, env_charge_coords, env_charges, env_charges_to_atoms_distances
    );
    return polarizable_embedding_energy_from_field(atomic_alphas, efield);
}
auto coulombic_embedding_energy(
    torch::Tensor atomic_charges,
    torch::Tensor env_charges,
    torch::Tensor env_charges_to_atoms_distances
) -> torch::Tensor {
    torch::Tensor pair_energies = torch::reshape(env_charges, {-1, 1}) *
        torch::reshape(atomic_charges, {1, -1}) / env_charges_to_atoms_distances;
    return torch::sum(pair_energies);
}
}  // namespace electro
