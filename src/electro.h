#include <torch/all.h>
#include <torch/script.h>

namespace electro {
auto polarizable_embedding_energy(
    torch::Tensor coords,
    torch::Tensor atomic_alphas,
    torch::Tensor env_charge_coords,
    torch::Tensor env_charges,
    torch::Tensor env_charges_to_atoms_distances,
    double inv_pol_dielectric
) -> torch::Tensor;
auto coulombic_embedding_energy(
    torch::Tensor atomic_charges,
    torch::Tensor env_charges,
    torch::Tensor env_charges_to_atoms_distances
) -> torch::Tensor;
}  // namespace electro
