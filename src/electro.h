#pragma once

#include <torch/all.h>

namespace electro {
auto polarizableEmbeddingEnergy(
    torch::Tensor ml_coords,
    torch::Tensor atomic_alphas,
    torch::Tensor mm_coords,
    torch::Tensor mm_atomic_charges,
    torch::Tensor mm_to_ml_distances,
    double inv_pol_dielectric
) -> torch::Tensor;
auto coulombicEmbeddingEnergy(
    torch::Tensor ml_atomic_charges,
    torch::Tensor mm_atomic_charges,
    torch::Tensor mm_to_ml_distances
) -> torch::Tensor;
}  // namespace electro
