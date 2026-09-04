#include <torch/all.h>
#include <torch/script.h>

namespace electro {
auto calc_efield(
    torch::Tensor coords,
    torch::Tensor env_charge_coords,
    torch::Tensor env_charges,
    torch::Tensor env_charges_to_atoms_distances
) -> torch::Tensor;
auto polarizable_embedding_energy_from_field(
    torch::Tensor atomic_alphas_bohr,
    torch::Tensor efield,
    double inv_pol_dielectric
) -> torch::Tensor;
auto convert_alphas_angstrom3_to_bohr3(
    torch::Tensor atomic_alphas
) -> torch::Tensor;
auto polarizable_embedding_energy(
    torch::Tensor coords,
    torch::Tensor atomic_alphas,
    torch::Tensor env_charge_coords,
    torch::Tensor env_charges,
    torch::Tensor env_charges_to_atoms_distances,
    double inv_pol_dielectric
) -> torch::Tensor;
auto polarizable_embedding_energy_with_bohr_alphas(
    torch::Tensor coords,
    torch::Tensor atomic_alphas_bohr,
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
