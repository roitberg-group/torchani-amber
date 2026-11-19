import math
import enum

import torch
from torch import Tensor

from torchani.units import HARTREE_TO_EV
from torchani.tuples import SpeciesEnergiesAtomicCharges


# NOTE: Placeholder implementation, not the actual polarizable model. The net charge
# is split evenly among the atoms, which have a fixed polarizability.
# Inputs are in Angstrom, e, V/Angstrom and V, output energy is in eV
class PolModel(torch.nn.Module):
    def __init__(self, polarizability: float = 0.01) -> None:
        super().__init__()
        self._polarizability = polarizability

    def forward(
        self,
        coord: Tensor,
        numbers: Tensor,
        charge: Tensor,
        mult: Tensor,
        efield: Tensor,
        potential: Tensor,
    ) -> dict[str, Tensor]:
        atomic_charges = charge.expand(numbers.shape[0]) / numbers.shape[0]
        coulomb = (atomic_charges * potential).sum()
        polarization = -0.5 * self._polarizability * efield.pow(2).sum()
        return {
            "energy": coulomb + polarization,
            "charges_eq_projected": atomic_charges,
        }

# Coulomb constant in V*A/e, same value used internally by PolModel
COULOMB_CONST_V_ANG = 14.3996

# Self atomic energies (eV)
# NOTE: These are placeholders, not the SAE of the PolModel model
SAE_EV = {
    1: -10.0,
    5: -10.0,
    6: -10.0,
    7: -10.0,
    8: -10.0,
    9: -10.0,
    14: -10.0,
    15: -10.0,
    16: -10.0,
    17: -10.0,
    33: -10.0,
    34: -10.0,
    35: -10.0,
    53: -10.0,
}

# GEEP widths (Angstrom) for H to Xe, indexed by atomic number - 1.
GEEP_SIGMAS_ANG: tuple[float, ...] = (
    1.00,
    1.20,
    1.60,
    1.50,
    1.50,
    1.50,
    1.40,
    1.30,
    1.25,
    1.30,  # H to Ne
    1.80,
    1.70,
    1.70,
    1.65,
    1.60,
    1.60,
    1.55,
    1.50,
    2.00,
    1.90,  # Na to Ca
    1.85,
    1.80,
    1.80,
    1.75,
    1.75,
    1.75,
    1.70,
    1.70,
    1.70,
    1.70,  # Sc to Zn
    1.75,
    1.70,
    1.70,
    1.70,
    1.70,
    1.70,
    2.10,
    2.00,
    1.95,
    1.90,  # Ga to Zr
    1.90,
    1.90,
    1.85,
    1.85,
    1.85,
    1.85,
    1.85,
    1.85,
    1.90,
    1.90,  # Nb to Sn
    1.90,
    1.90,
    1.85,
    1.80,  # Sb to Xe
)
CARBON_GEEP_SIGMA_ANG = GEEP_SIGMAS_ANG[5]

# Closed shell singlet
DEFAULT_MULT = 1.0

# Same default as the one used for AimNet2 DSF, in Angstrom^-1
DEFAULT_DSF_ALPHA = 0.2

# NOTE: In the original C++ code the modes are:
# - Min-image, hard cutoff, spherical (for ME, legacy, what QM/MM does in sander)
# - Min-image, no cutoff (for EE, current code does this, field is non-continuous)
# - Full-Ewald/PME (for ME, current code does this)


class CutoffKind(enum.Enum):
    HARD = "hard"
    DSF = "dsf"
    SWITCHING = "switching"


class _PolModelMLMM(torch.nn.Module):
    r"""PolModel with the ML/MM coupling computed internally, through compute_mlmm

    The MM charges are converted into the external potential and field at each ML atom,
    and the model computes the embedded energy (including the coupling) internally.
    Subclasses define the potential kernel of a single unit MM charge. Only a single
    conformation is supported.

    The kernel is modified according to cutoff_kind:

    - "hard": unmodified
    - "dsf": damped by erfc(dsf_alpha * r), then shifted so that both the potential
      and the field are zero at the cutoff (Fennell and Gezelter, 2006)
    - "switching": multiplied by a quintic switch that goes smoothly from 1 at
      switch_on to 0 at the cutoff

    The cutoff is the mm_cutoff argument of compute_mlmm, which is required for "dsf"
    and "switching", and must be > switch_on for "switching". With these, MM atoms
    beyond the cutoff contribute zero. With "hard" all MM atoms that are passed in
    contribute, and mm_cutoff has no effect.

    If scale_pairs is True, the (ml_idx, mm_idx) pairs passed as pairs_12 and pairs_13
    to compute_mlmm don't interact, and the pairs_14 interactions are scaled by 0.5.
    """

    _znum_to_sae: Tensor

    def __init__(
        self,
        cutoff_kind: CutoffKind | str = CutoffKind.HARD,
        dsf_alpha: float | None = None,
        switch_on: float | None = None,
        scale_pairs: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        cutoff_kind = CutoffKind(cutoff_kind)
        if cutoff_kind is CutoffKind.DSF:
            if dsf_alpha is None:
                dsf_alpha = DEFAULT_DSF_ALPHA
        elif dsf_alpha is not None:
            raise ValueError("dsf_alpha only has an effect for 'dsf'")
        if cutoff_kind is CutoffKind.SWITCHING:
            if switch_on is None:
                raise ValueError("switch_on is required for 'switching'")
            if switch_on < 0.0:
                raise ValueError("switch_on must be >= 0")
        elif switch_on is not None:
            raise ValueError("switch_on only has an effect for 'switching'")
        self._cutoff_kind = cutoff_kind.value
        self._dsf_alpha = 0.0 if dsf_alpha is None else dsf_alpha
        self._switch_on = 0.0 if switch_on is None else switch_on
        self._scale_pairs = scale_pairs

        self.model = PolModel(**kwargs)
        # Indexed by atomic number. Elements with no SAE are NaN, so that they can't
        # silently contribute zero to the energy
        znum_to_sae = torch.full((max(SAE_EV) + 1,), float("nan"), dtype=torch.double)
        for znum, sae in SAE_EV.items():
            znum_to_sae[znum] = sae
        self.register_buffer("_znum_to_sae", znum_to_sae)
        self._coulomb_const = COULOMB_CONST_V_ANG
        self._mult = DEFAULT_MULT
        self._hartree_to_ev = HARTREE_TO_EV

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
        species, coords = species_coordinates
        efield = coords.new_zeros(coords.shape)
        potential = coords.new_zeros(species.shape)
        energies, atomic_charges = self._run_model(
            species, coords, efield, potential, charge, None
        )
        return SpeciesEnergiesAtomicCharges(species, energies, atomic_charges)

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
        pairs_12: Tensor | None = None,
        pairs_13: Tensor | None = None,
        pairs_14: Tensor | None = None,
        atomic: bool = False,
        ensemble_values: bool = False,
        molecule_idxs: Tensor | None = None,
    ) -> Tensor:
        if atomic:
            raise ValueError("PolModel does not support atomic energies")
        species, coords = species_coordinates
        if mm_cutoff is None:
            if self._cutoff_kind != "hard":
                raise ValueError(f"'{self._cutoff_kind}' requires mm_cutoff")
            mm_cutoff = coords.new_zeros(coords.shape[0])
        # Shape: (C, 1, 1), to broadcast with the distances
        cutoff = mm_cutoff.view(-1, 1, 1).to(coords.dtype)
        # Shapes: delta (C, N, M, 3), distances (C, N, M)
        delta = coords.unsqueeze(2) - mm_coords.unsqueeze(1)
        distances = torch.linalg.norm(delta, dim=-1)
        pair_charges = self._pair_charges(
            mm_charges, species.shape[1], pairs_12, pairs_13, pairs_14
        )
        potential, efield = self._potential_and_efield(
            species, delta, distances, pair_charges, cutoff
        )
        energies, _ = self._run_model(
            species, coords, efield, potential, charge, spin_mult
        )
        return energies

    # Output shape: (C, N, M) if the pairs are scaled, otherwise (C, 1, M)
    def _pair_charges(
        self,
        mm_charges: Tensor,
        ml_atoms_num: int,
        pairs_12: Tensor | None,
        pairs_13: Tensor | None,
        pairs_14: Tensor | None,
    ) -> Tensor:
        pair_charges = mm_charges.unsqueeze(1)
        if not self._scale_pairs:
            return pair_charges
        pair_scale = mm_charges.new_ones((ml_atoms_num, mm_charges.shape[1]))
        if pairs_14 is not None:
            pair_scale[pairs_14[0], pairs_14[1]] = 0.5
        if pairs_12 is not None:
            pair_scale[pairs_12[0], pairs_12[1]] = 0.0
        if pairs_13 is not None:
            pair_scale[pairs_13[0], pairs_13[1]] = 0.0
        return pair_charges * pair_scale

    # Output shapes: potential (C, N) in V, efield (C, N, 3) in V/Angstrom
    def _potential_and_efield(
        self,
        species: Tensor,
        delta: Tensor,
        distances: Tensor,
        pair_charges: Tensor,
        cutoff: Tensor,
    ) -> tuple[Tensor, Tensor]:
        ker, ker_deriv = self._cut_kernel(species, distances, cutoff)
        potential = self._coulomb_const * (pair_charges * ker).sum(-1)
        # E = -grad(V) wrt the ML coords, and grad(r) = delta / r
        pair_efield = -pair_charges * ker_deriv / distances
        efield = self._coulomb_const * (pair_efield.unsqueeze(-1) * delta).sum(2)
        return potential, efield

    def _cut_kernel(
        self, species: Tensor, distances: Tensor, cutoff: Tensor
    ) -> tuple[Tensor, Tensor]:
        ker, ker_deriv = self._kernel(species, distances)
        if self._cutoff_kind == "hard":
            return ker, ker_deriv
        if self._cutoff_kind == "switching":
            return self._switch(ker, ker_deriv, distances, cutoff)
        # DSF branch
        # Dampoing
        ker, ker_deriv = self._damp(ker, ker_deriv, distances)

        # Shifting
        ker_at_cut, deriv_at_cut = self._kernel(species, cutoff)
        ker_at_cut, deriv_at_cut = self._damp(ker_at_cut, deriv_at_cut, cutoff)
        shifted_ker = ker - ker_at_cut - deriv_at_cut * (distances - cutoff)

        # Clamping
        is_inside = distances < cutoff
        zeros = torch.zeros_like(distances)
        return (
            torch.where(is_inside, shifted_ker, zeros),
            torch.where(is_inside, ker_deriv - deriv_at_cut, zeros),
        )

    def _damp(
        self, ker: Tensor, ker_deriv: Tensor, distances: Tensor
    ) -> tuple[Tensor, Tensor]:
        damping = torch.erfc(self._dsf_alpha * distances)
        damping_deriv = (
            -2.0
            * self._dsf_alpha
            / math.sqrt(math.pi)
            * torch.exp(-((self._dsf_alpha * distances) ** 2))
        )
        return ker * damping, ker_deriv * damping + ker * damping_deriv

    def _switch(
        self, ker: Tensor, ker_deriv: Tensor, distances: Tensor, cutoff: Tensor
    ) -> tuple[Tensor, Tensor]:
        width = cutoff - self._switch_on
        x = ((distances - self._switch_on) / width).clamp(0.0, 1.0)
        switch = 1.0 - 10.0 * x**3 + 15.0 * x**4 - 6.0 * x**5
        switch_deriv = -30.0 * x**2 * (1.0 - x) ** 2 / width
        return ker * switch, ker_deriv * switch + ker * switch_deriv

    # Potential of a unit MM charge (in units of the Coulomb constant), and its
    # derivative wrt the distance. Shapes: (C, N, M)
    def _kernel(self, species: Tensor, distances: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError("Must be overriden by subclasses")

    # Inputs are in Angstrom, V/Angstrom, V and e, output is in Hartree
    def _run_model(
        self,
        species: Tensor,
        coords: Tensor,
        efield: Tensor,
        potential: Tensor,
        charge: Tensor | None,
        spin_mult: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if coords.shape[0] != 1:
            raise ValueError("PolModel only supports a single conformation")
        if charge is None:
            charge = coords.new_zeros(1)
        if spin_mult is None:
            spin_mult = coords.new_full((1,), self._mult)
        out = self.model(
            coords.squeeze(0),
            species.squeeze(0).long(),
            charge.to(coords.dtype).view(1),
            spin_mult.to(coords.dtype).view(1),
            efield.squeeze(0),
            potential.squeeze(0),
        )
        sae = self._znum_to_sae[species].sum(-1)
        energies = (out["energy"].view(1).double() + sae) / self._hartree_to_ev
        if "charges_eq_projected" in out:
            atomic_charges = out["charges_eq_projected"].view(1, -1)
        else:
            atomic_charges = coords.new_empty((1, 0))
        return energies, atomic_charges


class PolModelCoulomb(_PolModelMLMM):
    r"""PolModel embedded in the potential and field of MM point charges"""

    def _kernel(self, species: Tensor, distances: Tensor) -> tuple[Tensor, Tensor]:
        return 1.0 / distances, -1.0 / distances**2


class PolModelGeep(_PolModelMLMM):
    r"""PolModel embedded in the potential and field of Gaussian MM charges (GEEP)

    The width of each Gaussian depends on the element of the ML atom, and is the same as
    the one used by pmemd. The arguments correspond to the geep_factor and geep_element
    entries of the pmemd &mlp namelist (use_element_sigma=False is geep_element=0).
    """

    _znum_to_sigma: Tensor

    def __init__(
        self,
        geep_factor: float = 1.0,
        use_element_sigma: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if geep_factor <= 0.0:
            raise ValueError("geep_factor must be > 0")
        if use_element_sigma:
            sigmas = GEEP_SIGMAS_ANG
        else:
            sigmas = (CARBON_GEEP_SIGMA_ANG,) * len(GEEP_SIGMAS_ANG)
        # Index 0 is a dummy so that the buffer can be indexed by atomic number
        znum_to_sigma = torch.tensor((math.nan,) + sigmas) * geep_factor
        self.register_buffer("_znum_to_sigma", znum_to_sigma)
        self._sqrt_2_over_pi = math.sqrt(2.0 / math.pi)

    def _kernel(self, species: Tensor, distances: Tensor) -> tuple[Tensor, Tensor]:
        # Shapes: sigma (C, N, 1), scaled_distances (C, N, M)
        sigma = self._znum_to_sigma[species].to(distances.dtype).unsqueeze(-1)
        scaled_distances = distances / (math.sqrt(2.0) * sigma)
        erf = torch.erf(scaled_distances)
        gaussian = torch.exp(-(scaled_distances**2))
        ker = erf / distances
        ker_deriv = -erf / distances**2 + self._sqrt_2_over_pi * gaussian / (
            sigma * distances
        )
        return ker, ker_deriv


def generate_polmodel_wrapper(path, use_geep: bool = False, **kwargs) -> None:
    if use_geep:
        model: _PolModelMLMM = PolModelGeep(**kwargs)
    else:
        model = PolModelCoulomb(**kwargs)
    torch.jit.save(torch.jit.script(model), path)
