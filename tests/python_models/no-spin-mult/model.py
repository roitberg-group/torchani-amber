r"""Python model without a spin_mult argument, used by the libtorchani tests

make_model returns the model in float32 and in training mode, so that the tests can
check that libtorchani moves it to the requested dtype and sets eval mode
"""

import torch
from torch import Tensor


class NoSpinMultModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.k = torch.nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        species_coordinates: tuple[Tensor, Tensor],
        cell: Tensor | None = None,
        pbc: Tensor | None = None,
        charge: Tensor | None = None,
        atomic: bool = False,
        ensemble_values: bool = False,
        molecule_idxs: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        species, coords = species_coordinates
        energies = self.k * (coords**2).sum((-1, -2))
        if self.training:
            energies = energies + 1000.0
        return species, energies


def make_model(device: str, dtype: torch.dtype) -> torch.nn.Module:
    return NoSpinMultModel().to(device=device, dtype=torch.float32).train()
