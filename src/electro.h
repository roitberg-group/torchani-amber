#pragma once

#include <torch/all.h>

namespace electro {
auto polarizableEmbeddingEnergy(
    torch::Tensor coords,
    torch::Tensor atomic_alphas,
    torch::Tensor env_charge_coords,
    torch::Tensor env_charges,
    torch::Tensor env_charges_to_atoms_distances,
    double inv_pol_dielectric
) -> torch::Tensor;
auto coulombicEmbeddingEnergy(
    torch::Tensor atomic_charges,
    torch::Tensor env_charges,
    torch::Tensor env_charges_to_atoms_distances
) -> torch::Tensor;
}  // namespace electro
