#include "electro.h"

#include <cmath>
#include <numbers>
#include <optional>

#include <torch/all.h>

#include "constants.h"

namespace {
// The GEEP kernels are finite at r = 0, but their formulas are 0 / 0 there, and lose
// precision close to it (mostly the field, in float32). Below this distance (Angstrom)
// the kernels take their value at it, which is close to their r = 0 limit
constexpr double GEEP_MIN_DISTANCE = 0.001;

// Potential of a unit Gaussian charge of width sigma, erf(x) / r, with
// x = r / (sqrt(2) * sigma). Output has the shape of mm_to_ml_distances
auto geepPotentialKernel(torch::Tensor ml_geep_sigmas, torch::Tensor mm_to_ml_distances)
    -> torch::Tensor {
    torch::Tensor distances = mm_to_ml_distances.clamp_min(GEEP_MIN_DISTANCE);
    torch::Tensor scaled_distances =
        distances / (std::sqrt(2.0) * ml_geep_sigmas.view({1, -1}));
    return torch::erf(scaled_distances) / distances;
}

// Field of a unit Gaussian charge of width sigma, divided by r (the field of a point
// charge is 1 / r^3 times the displacement), (erf(x) - sqrt(2 / pi) * (r / sigma) *
// exp(-x^2)) / r^3, with x = r / (sqrt(2) * sigma)
auto geepEfieldKernel(torch::Tensor ml_geep_sigmas, torch::Tensor mm_to_ml_distances)
    -> torch::Tensor {
    torch::Tensor distances = mm_to_ml_distances.clamp_min(GEEP_MIN_DISTANCE);
    torch::Tensor sigmas = ml_geep_sigmas.view({1, -1});
    torch::Tensor scaled_distances = distances / (std::sqrt(2.0) * sigmas);
    torch::Tensor screening = torch::erf(scaled_distances)
        - std::sqrt(2.0 / std::numbers::pi) * distances / sigmas
            * torch::exp(-scaled_distances.pow(2));
    return screening / distances.pow(3);
}

// NOTE: Without GEEP, the point charge kernels (here and in coulombicEmbeddingEnergy)
// diverge, and give inf or NaN for very small mm_to_ml_distances
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
 * ml_geep_sigmas: shape (ml_atoms_num,)
 */
auto calcEfield(
    torch::Tensor ml_coords,
    torch::Tensor mm_coords,
    torch::Tensor mm_atomic_charges,
    torch::Tensor mm_to_ml_distances,
    std::optional<torch::Tensor> ml_geep_sigmas
) -> torch::Tensor {
    auto delta_coords =
        torch::reshape(mm_coords, {-1, 1, 3}) - torch::reshape(ml_coords, {1, -1, 3});
    torch::Tensor kernel;
    if (ml_geep_sigmas) {
        kernel = geepEfieldKernel(*ml_geep_sigmas, mm_to_ml_distances);
    } else {
        kernel = 1.0 / mm_to_ml_distances.pow(3);
    }
    auto efield = torch::reshape(mm_atomic_charges, {-1, 1, 1}) * delta_coords
        * kernel.unsqueeze(-1);
    // Inputs are Ang for *coords, and e for mm_atomic_charges
    // Electric fields with angstrom units are unintuitive, to avoid that
    // Output in AU (Hartree / e), so perform the conversion:
    // E_field_AU = E_field_Angstrom / A2B**2
    return torch::sum(efield, 0) / std::pow(constants::CODATA06_ANGSTROM_TO_BOHR, 2);
}

/**
 * Given efield and alphas, calculate the associated QM/MM energy
 */
auto polarizableEmbeddingEnergyFromField(
    torch::Tensor atomic_alphas, torch::Tensor efield, double inv_pol_dielectric
) -> torch::Tensor {
    // Input atomic alphas and Efield are in AU
    return -inv_pol_dielectric
        * torch::sum(atomic_alphas * torch::sum(efield.pow(2), 1));
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
    double inv_pol_dielectric,
    std::optional<torch::Tensor> ml_geep_sigmas
) -> torch::Tensor {
    torch::Tensor efield = calcEfield(
        ml_coords, mm_coords, mm_atomic_charges, mm_to_ml_distances, ml_geep_sigmas
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
    torch::Tensor mm_to_ml_distances,
    std::optional<torch::Tensor> ml_geep_sigmas
) -> torch::Tensor {
    torch::Tensor kernel;
    if (ml_geep_sigmas) {
        kernel = geepPotentialKernel(*ml_geep_sigmas, mm_to_ml_distances);
    } else {
        kernel = 1.0 / mm_to_ml_distances;
    }
    torch::Tensor pair_energies = torch::reshape(mm_atomic_charges, {-1, 1})
        * torch::reshape(ml_atomic_charges, {1, -1}) * kernel;
    // Output in AU (Hartree), so perform the conversion
    // Epot_AU = E_pot_Angstrom / A2B
    return torch::sum(pair_energies) / constants::CODATA06_ANGSTROM_TO_BOHR;
}
}  // namespace electro
