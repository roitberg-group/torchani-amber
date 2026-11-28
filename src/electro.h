#pragma once

#include <optional>

#include <torch/all.h>

namespace electro {
// If ml_geep_sigmas (Angstrom, shape (ml_atoms_num,)) is passed, the MM charges are
// Gaussians (GEEP), with the width of the ML atom they interact with
auto polarizableEmbeddingEnergy(
    torch::Tensor ml_coords,
    torch::Tensor atomic_alphas,
    torch::Tensor mm_coords,
    torch::Tensor mm_atomic_charges,
    torch::Tensor mm_to_ml_distances,
    double inv_pol_dielectric,
    std::optional<torch::Tensor> ml_geep_sigmas = std::nullopt
) -> torch::Tensor;
auto coulombicEmbeddingEnergy(
    torch::Tensor ml_atomic_charges,
    torch::Tensor mm_atomic_charges,
    torch::Tensor mm_to_ml_distances,
    std::optional<torch::Tensor> ml_geep_sigmas = std::nullopt
) -> torch::Tensor;
}  // namespace electro
