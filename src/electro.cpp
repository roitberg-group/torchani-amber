#include <torch/all.h>
#include <torch/script.h>

namespace electro {
/**
 * Compute the distance matrix between the coordinates of charges and atoms
 * This function scales as num-atoms x num-charges
 *
 * coords: shape (num-atoms, 3)
 * charge_coords: shape (num-charges, 3)
 *
 * returns: norm_delta_coords: shape (num_atoms, num_charges)
 */
auto norm_delta_coords(torch::Tensor charge_coords, torch::Tensor coords)
    -> torch::Tensor {
    auto delta_coords =
        torch::reshape(charge_coords, {-1, 1, 3}) - torch::reshape(coords, {1, -1, 3});
    auto norm_delta_coords = torch::linalg_norm(delta_coords, std::nullopt, 2, true);
    return norm_delta_coords;
}

/**
 * Calculate the electric field that a sequence of charges, located at
 * 'charge_coords', on some other 'coords'. Shape of 'coords' and
 * 'charge_coords' doesn't need to match, but first dim of 'charge_coords' and
 * 'charges' must match
 *
 * coords: shape (num_atoms, 3)
 * charge_coords: shape (num_charges, 3)
 * charges: shape (num_charges,)
 */
auto efield(
    torch::Tensor charges,
    torch::Tensor charge_coords,
    torch::Tensor coords,
    torch::Tensor norm_delta_coords
) -> torch::Tensor {
    auto delta_coords =
        torch::reshape(charge_coords, {-1, 1, 3}) - torch::reshape(coords, {1, -1, 3});
    auto efield =
        torch::reshape(charges, {-1, 1, 1}) * delta_coords / norm_delta_coords.pow(3);
    return torch::sum(efield, 0);
}
}  // namespace electro
