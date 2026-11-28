r"""Python model whose 'forward' takes 'charge: int', used by the libtorchani tests

The energy is sum(coords**2) + charge, and the model raises if charge is not an int
"""

import torch
from torch import Tensor


class LegacyChargeModel(torch.nn.Module):
    def forward(
        self,
        species_coordinates: tuple[Tensor, Tensor],
        cell: Tensor | None = None,
        pbc: Tensor | None = None,
        charge: int = 0,
        atomic: bool = False,
        ensemble_values: bool = False,
        molecule_idxs: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if not isinstance(charge, int):
            raise TypeError(f"Expected an int charge, got {type(charge)}")
        species, coords = species_coordinates
        return species, (coords**2).sum((-1, -2)) + charge


def make_model(device: str, dtype: torch.dtype) -> torch.nn.Module:
    return LegacyChargeModel()
