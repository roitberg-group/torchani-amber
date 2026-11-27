import torch
from torch import Tensor


def hydrogen_self_energy(num: Tensor) -> Tensor:
    return -0.5 * num


def oxygen_self_energy(num: Tensor) -> Tensor:
    return -75.0 * num


class ToyModel(torch.nn.Module):
    r"""Pairwise Morse-like potential that is neither scriptable nor AOTI-compilable

    The dict of per-element functions can't be scripted, and the .tolist() call makes the
    control flow depend on the input data.
    """

    def __init__(self) -> None:
        super().__init__()
        self.depth = torch.nn.Parameter(torch.tensor(0.01, dtype=torch.double))
        self.self_energy = {1: hydrogen_self_energy, 8: oxygen_self_energy}

    def forward(self, species_coordinates: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        species, coords = species_coordinates
        energies = coords.new_zeros(species.shape[0])
        for z in species.unique().tolist():
            energies = energies + self.self_energy[z]((species == z).sum(-1))
        dists = torch.cdist(coords, coords)
        pair_is_valid = ~torch.eye(species.shape[1], dtype=torch.bool)
        r = dists[:, pair_is_valid]
        morse = self.depth * (1 - torch.exp(-(r - 1.0))) ** 2
        energies = energies + 0.5 * morse.sum(-1)
        return species, energies


def make_model(device: str, dtype: torch.dtype) -> torch.nn.Module:
    return ToyModel().to(device=device, dtype=dtype)
