import torch
from torch import Tensor

import torchani
from torchani.tuples import SpeciesEnergiesAtomicCharges

# Same constant as src/electro.cpp
ANGSTROM_TO_BOHR = 1.8897261328727875

# Default atomic polarizabilities in AU (H to Ar), and default inverse dielectric.
# Same as in AmberTools/src/sander/qm2_extern_torchani_module.F90 (amber source)
DEFAULT_ALPHAS_AU = (
    3.08,
    1.38375,
    164.1125,
    37.74,
    20.5,
    11.3,
    7.4,
    5.3,
    3.74,
    2.6611,
    162.7,
    71.2,
    57.8,
    37.3,
    25.0,
    19.4,
    14.6,
    11.083,
)
DEFAULT_INV_POL_DIELECTRIC = 0.5


class CustomMLMMANImbis(torch.nn.Module):
    r"""ANImbis with the ML/MM coupling computed internally, through compute_mlmm

    The coupling energy is the same as the one computed by torchani-amber with
    mlmm_coupling='simple-polarizable' and use_mlip_atomic_charges=.true., with the
    default polarizabilities and polarization dielectric.
    """

    _znum_to_alpha: Tensor

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.model = torchani.models.ANImbis(**kwargs)
        # Index 0 is a dummy so that the buffer can be indexed by atomic number.
        alphas_au = torch.tensor((0.0,) + DEFAULT_ALPHAS_AU, dtype=torch.double)
        self.register_buffer("_znum_to_alpha", alphas_au)
        self._inv_pol_dielectric = DEFAULT_INV_POL_DIELECTRIC
        self._angstrom_to_bohr = ANGSTROM_TO_BOHR
        self.cutoff = self.model.cutoff

    @property
    @torch.jit.unused
    def symbols(self) -> tuple[str, ...]:
        return self.model.symbols

    @torch.jit.export
    def set_active_members(self, idxs: list[int]) -> None:
        self.model.set_active_members(idxs)

    @torch.jit.export
    def set_strategy(self, strategy: str) -> None:
        self.model.set_strategy(strategy)

    def forward(
        self,
        species_coordinates: tuple[Tensor, Tensor],
        cell: Tensor | None = None,
        pbc: Tensor | None = None,
        charge: Tensor | None = None,
        atomic: bool = False,
        ensemble_values: bool = False,
        molecule_idxs: Tensor | None = None,
    ) -> SpeciesEnergiesAtomicCharges:
        return self.model(
            species_coordinates,
            cell=cell,
            pbc=pbc,
            charge=charge,
            atomic=atomic,
            ensemble_values=ensemble_values,
            molecule_idxs=molecule_idxs,
        )

    # Inputs are in Angstrom and e, output is in Hartree
    @torch.jit.export
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
        if spin_mult is not None and bool((spin_mult != 1).any()):
            raise ValueError("The custom ML/MM wrapper only supports spin_mult = 1")
        output = self.model(
            species_coordinates,
            cell=cell,
            pbc=pbc,
            charge=charge,
            atomic=atomic,
            ensemble_values=ensemble_values,
            molecule_idxs=molecule_idxs,
        )
        # Shapes: delta (C, M, N, 3), distances (C, M, N), efield (C, N, 3)
        delta = mm_coords.unsqueeze(2) - coords.unsqueeze(1)
        distances = torch.linalg.norm(delta, dim=-1)
        pair_charges = mm_charges.unsqueeze(2)
        pair_coulomb = pair_charges * output.atomic_charges.unsqueeze(1) / distances
        atomic_coulomb = pair_coulomb.sum(1) / self._angstrom_to_bohr

        pair_efield = pair_charges.unsqueeze(-1) * delta / distances.unsqueeze(-1) ** 3
        efield = pair_efield.sum(1) / self._angstrom_to_bohr**2
        alphas = self._znum_to_alpha[species]
        atomic_pol = -self._inv_pol_dielectric * alphas * efield.pow(2).sum(-1)
        atomic_dist = -0.5 * atomic_pol

        atomic_coupling = atomic_dist + atomic_pol + atomic_coulomb
        if atomic:
            return output.energies + atomic_coupling
        return output.energies + atomic_coupling.sum(-1)


def generate_custom_mlmm_wrapper(path, **kwargs) -> None:
    torch.jit.save(torch.jit.script(CustomMLMMANImbis(**kwargs)), path)
