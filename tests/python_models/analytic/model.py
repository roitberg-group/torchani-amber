r"""Python model with analytic energies, used by the libtorchani tests

Energies are in Hartree, per molecule:

forward:
    K * sum(coords**2) + CHARGE_SCALE * charge + SPIN_SCALE * spin_mult
    + sum(extra_features)
compute_from_external_neighbors:
    K * sum(coords**2) + PAIR_SCALE * num_pairs
compute_mlmm:
    K * sum(ml_coords**2) + sum(mm_charges * sum(mm_coords**2, -1))
compute_multipole_mlmm:
    The same as compute_mlmm, plus DIPOLE_SCALE * sum(mm_dipoles), and it also returns
    atomic charges equal to the atomic numbers
"""

import torch
from torch import Tensor

K = 0.5
CHARGE_SCALE = 0.1
SPIN_SCALE = 0.01
PAIR_SCALE = 0.001
DIPOLE_SCALE = 0.2


class AnalyticModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.k = torch.nn.Parameter(torch.tensor(K))
        self.active_members: list[int] = []
        self.extra_features: list[float] = []
        self.strategy = "pyaev"

    def set_active_members(self, idxs: list[int]) -> None:
        self.active_members = idxs

    def set_extra_features(self, features: list[float]) -> None:
        self.extra_features = features

    def set_strategy(self, strategy: str) -> None:
        if strategy not in ("pyaev", "cuaev"):
            raise ValueError(f"Unsupported strategy {strategy}")
        self.strategy = strategy

    def _ml_energies(self, coords: Tensor) -> Tensor:
        return self.k * (coords**2).sum((-1, -2))

    def forward(
        self,
        species_coordinates: tuple[Tensor, Tensor],
        cell: Tensor | None = None,
        pbc: Tensor | None = None,
        charge: Tensor | None = None,
        spin_mult: Tensor | None = None,
        atomic: bool = False,
        ensemble_values: bool = False,
        molecule_idxs: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        species, coords = species_coordinates
        energies = self._ml_energies(coords) + sum(self.extra_features)
        if charge is not None:
            energies = energies + CHARGE_SCALE * charge.to(coords.dtype)
        if spin_mult is not None:
            energies = energies + SPIN_SCALE * spin_mult.to(coords.dtype)
        return species, energies

    def compute_from_external_neighbors(
        self,
        species: Tensor,
        coords: Tensor,
        neighbor_idxs: Tensor,
        shifts: Tensor,
        charge: Tensor | None = None,
        spin_mult: Tensor | None = None,
        atomic: bool = False,
        ensemble_values: bool = False,
        molecule_idxs: Tensor | None = None,
    ) -> dict[str, Tensor]:
        energies = self._ml_energies(coords) + PAIR_SCALE * neighbor_idxs.shape[1]
        return {"energies": energies}

    def compute_mlmm(
        self,
        species_coordinates: tuple[Tensor, Tensor],
        mm_coords: Tensor,
        mm_charges: Tensor,
        cell: Tensor | None = None,
        pbc: Tensor | None = None,
        charge: Tensor | None = None,
        spin_mult: Tensor | None = None,
        mm_cutoff: Tensor | None = None,
        atomic: bool = False,
        ensemble_values: bool = False,
        molecule_idxs: Tensor | None = None,
    ) -> Tensor:
        species, coords = species_coordinates
        mm_energies = (mm_charges * (mm_coords**2).sum(-1)).sum(-1)
        return self._ml_energies(coords) + mm_energies

    def compute_multipole_mlmm(
        self,
        species_coordinates: tuple[Tensor, Tensor],
        mm_coords: Tensor,
        mm_charges: Tensor,
        mm_dipoles: Tensor,
        mm_quads: Tensor,
        cell: Tensor | None = None,
        pbc: Tensor | None = None,
        charge: Tensor | None = None,
        spin_mult: Tensor | None = None,
        mm_cutoff: Tensor | None = None,
        atomic: bool = False,
        ensemble_values: bool = False,
        molecule_idxs: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        species, coords = species_coordinates
        energies = self.compute_mlmm(species_coordinates, mm_coords, mm_charges)
        energies = energies + DIPOLE_SCALE * mm_dipoles.sum()
        return energies, species.to(coords.dtype)


def make_model(device: str, dtype: torch.dtype) -> torch.nn.Module:
    return AnalyticModel().to(device=device, dtype=dtype)
