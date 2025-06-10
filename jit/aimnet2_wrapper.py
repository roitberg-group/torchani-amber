import dataclasses
import warnings
from dataclasses import asdict
from pathlib import Path
import typing_extensions as tpx
import typing as tp

import ase
import torch
from torch import Tensor
from torch.nn import Module

from torchani._testing import make_molec
from torchani.electro import ChargeNormalizer
from torchani.units import HARTREE_TO_EV
from torchani.utils import map_to_central
from torchani.tuples import SpeciesEnergiesAtomicCharges
from torchani.arch import _fetch_state_dict
from torchani.neighbors import (
    AdaptiveList,
    discard_outside_cutoff,
    discard_inter_molecule_pairs,
    reconstruct_shifts,
    Neighbors,
)
from torchani.arch import Assembler, ANIq
from torchani.utils import SYMBOLS_2X
from torchani.nn._internal import _ANINetworksDiscardFirstScalar, _ZeroANINetworks
try:
    from aimnet2calc.aimnet2ase import AIMNet2ASE
    AIMNET2CALC_AVAIL = True
except ImportError:
    AIMNET2CALC_AVAIL = True
    warnings.warn("Disabling tests. Install aimnet2calc for testing")


@dataclasses.dataclass
class AimNetArgs:
    method_lr: str
    cutoff_lr: tp.Optional[float] = None
    dsf_alpha: tp.Optional[float] = None
    output_hartree: bool = True

    @property
    def suffix(self) -> str:
        return self.method_lr.replace("simple", "nocut")


# Basically the only important parameter here is method_lr. It is better to leave the
# other params as their defaults NOTE: All networks support all methods, and they can be
# changed internally with no issue
class AimNet2Wrapper(torch.nn.Module):
    def __init__(
        self,
        model: Module,
        method_lr: tp.Optional[str] = None,
        cutoff_lr: tp.Optional[float] = None,
        dsf_alpha: tp.Optional[float] = None,
        output_hartree: bool = True,
    ) -> None:
        super().__init__()
        if not hasattr(model, "cutoff_lr"):
            raise ValueError("Only AimNet2 models with 'lr' are currently supported")

        # If method is not passed, fetch the network's default method
        if method_lr is None:
            for name, module in model.named_modules():
                if name.endswith(".lrcoulomb"):
                    method_lr = module.method

        # DSF = Damped Shifted Force, both the potential and the force go smoothly to 0
        if method_lr not in ("simple", "dsf", "ewald"):
            raise ValueError(f"Invalid lr method {method_lr}")
        if method_lr == "simple":
            if cutoff_lr is not None and cutoff_lr != float("inf"):
                raise ValueError("cutoff_lr not allowed for 'simple' lr")
            cutoff_lr = float("inf")
        elif cutoff_lr is None:
            cutoff_lr = 15.0  # default cutoff for both "dsf" and "ewald"
        if method_lr == "dsf":
            if dsf_alpha is None:
                dsf_alpha = 0.2  # default alpha for "dsf"
        elif dsf_alpha is not None:
            raise ValueError("dsf_alpha only has an effect for DSF")

        for name, module in model.named_modules():
            if name.endswith(".lrcoulomb"):
                module.method = method_lr
                if dsf_alpha is not None:
                    module.dsf_alpha = dsf_alpha
                    # NOTE: module.cutoff_lr = cutoff_lr is not done in the original
                    # AimNet code. Doing it seems to have no effect

        device = next(iter(model.parameters())).device
        dtype = next(iter(model.parameters())).dtype
        self._model = model
        self._nlist = AdaptiveList().to(device=device, dtype=dtype)
        if cutoff_lr < model.cutoff:
            raise ValueError("cutoff_lr must be < cutoff")
        self._cutoff = model.cutoff
        self._cutoff_lr = cutoff_lr
        self._factor = 1.0 / HARTREE_TO_EV if output_hartree else 1.0

    @classmethod
    def from_jit_file(
        cls,
        path: tp.Union[Path, str],
        device: str = "cpu",
        method_lr: tp.Optional[str] = None,
        cutoff_lr: tp.Optional[float] = None,
        dsf_alpha: tp.Optional[float] = None,
        output_hartree: bool = True,
    ) -> tpx.Self:
        path = Path(path)
        model = torch.jit.load(path, map_location=device)
        return cls(model, method_lr, cutoff_lr, dsf_alpha, output_hartree)

    def forward(
        self,
        species_coords: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        charge: int = 0,
        atomic: bool = False,
        ensemble_values: bool = False,
        _molecule_idxs: tp.Optional[Tensor] = None,
    ) -> SpeciesEnergiesAtomicCharges:
        species, coords = species_coords
        neighbors_lr = self._calc_neighbors(species, coords, cell, pbc, _molecule_idxs)
        # Delegate call since torchscript does not support super().forward()
        return self._run_aimnet_forward(
            species,
            coords,
            neighbors_lr,
            cell,
            pbc,
            charge,
            atomic,
            ensemble_values,
        )

    def _run_aimnet_forward(
        self,
        species: Tensor,
        coords: Tensor,
        neighbors_lr: Neighbors,
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        charge: int = 0,
        atomic: bool = False,
        ensemble_values: bool = False,
    ) -> SpeciesEnergiesAtomicCharges:

        neigh = discard_outside_cutoff(neighbors_lr, self._cutoff)
        atoms_num = coords.shape[1]
        data: tp.Dict[str, Tensor] = {
            "mol_idx": coords.new_zeros(atoms_num + 1),
            "numbers": self.pad_dim0(species.squeeze(0)).long(),
            "charge": coords.new_zeros(1),
            "cutoff_lr": coords.new_full((1,), fill_value=self._cutoff_lr),
        }
        if pbc is None:
            nbmat_lr, _, _, _ = self._sparse_to_dense(neighbors_lr.indices, atoms_num)
            nbmat, _, _, _ = self._sparse_to_dense(neigh.indices, atoms_num)
        else:
            assert cell is not None
            inv_cell = torch.inverse(cell)
            # TODO: It is inefficient to do this twice, it can be improved, but it is
            # probably not worth it for now
            # TODO: Check exactly what to detach
            nbmat_lr, shifts_lr, mask_lr = self._sparse_to_dense_pbc(
                coords.detach(), neighbors_lr, inv_cell.detach()
            )
            nbmat, shifts, mask = self._sparse_to_dense_pbc(
                coords.detach(), neigh, inv_cell.detach()
            )
            # Map coords to central cell
            coords = map_to_central(coords, cell, pbc)
            data["cell"] = cell
            data["shifts"] = shifts
            data["shifts_lr"] = shifts_lr
            data["nb_pad_mask"] = ~mask
            data["nb_pad_mask_lr"] = ~mask_lr
        data["coord"] = self.pad_dim0(coords.squeeze(0))
        data["nbmat"] = nbmat
        data["nbmat_lr"] = nbmat_lr
        data["mult"] = coords.new_ones(1)

        out = self._model(data)
        energy = out["energy"].view(coords.size(0))
        # Last charge for each molecule is the total charge
        atomic_charges = out["charges"].view(coords.size(0), coords.size(1) + 1)[:, :-1]
        return SpeciesEnergiesAtomicCharges(
            species, energy * self._factor, atomic_charges
        )

    @staticmethod
    def pad_dim0(a: Tensor) -> Tensor:
        # Quirky padding required by AimNet
        shapes = [0] * ((a.ndim - 1) * 2) + [0, 1]
        return torch.nn.functional.pad(a, shapes, mode="constant", value=0.0)

    def _sparse_to_dense_pbc(
        self, coords: Tensor, neighbors: Neighbors, inv_cell: Tensor
    ) -> tp.Tuple[Tensor, Tensor, Tensor]:
        atoms_num = coords.shape[1]
        shift_idxs = torch.round(reconstruct_shifts(coords, neighbors) @ inv_cell)
        dense_nb, sort_idxs, mask, max_neigh = self._sparse_to_dense(
            neighbors.indices, atoms_num
        )
        full_shiftidxs = -torch.cat((shift_idxs, -shift_idxs), dim=0)
        dense_shifts = shift_idxs.new_full((atoms_num + 1, max_neigh, 3), fill_value=-1)
        dense_shifts.masked_scatter_(mask.unsqueeze(-1), full_shiftidxs[sort_idxs])
        return dense_nb, dense_shifts, mask

    def _sparse_to_dense(
        self, idx: Tensor, atoms_num: int
    ) -> tp.Tuple[Tensor, Tensor, Tensor, int]:
        num_neighbors = torch.bincount(idx.view(-1), minlength=atoms_num + 1)
        max_neigh = int(num_neighbors.max().item())
        mask = (
            num_neighbors.unsqueeze(1)
            > torch.arange(max_neigh, device=idx.device, dtype=idx.dtype)
        ).to(torch.bool)
        full_idx = torch.cat((idx, idx.flipud()), dim=1)
        sort_idxs = torch.sort(full_idx[0]).indices
        # This +1 is not really useful but is added for compatibility
        dense_nb = idx.new_full((atoms_num + 1, max_neigh), fill_value=atoms_num)
        dense_nb.masked_scatter_(mask, full_idx[1, sort_idxs])
        return dense_nb, sort_idxs, mask, max_neigh

    def _calc_neighbors(
        self,
        species: Tensor,
        coords: Tensor,
        cell: tp.Optional[Tensor],
        pbc: tp.Optional[Tensor],
        _molecule_idxs: tp.Optional[Tensor],
    ) -> Neighbors:
        neigh_lr = self._nlist(self._cutoff_lr, species, coords.detach(), cell, pbc)
        # Experimental _molecule_idxs feature
        if _molecule_idxs is not None:
            if not (torch.jit.is_scripting() or torch.compiler.is_compiling()):
                warnings.warn("molecule_idxs is experimental and subject to change")
            if coords.shape[0] != 1:
                raise ValueError("molecule_idxs expects only one conformation")
            if len(_molecule_idxs) != coords.shape[1]:
                raise ValueError(
                    "molecule_idxs must be the same length as num atoms, if passed"
                )
            neigh_lr = discard_inter_molecule_pairs(neigh_lr, _molecule_idxs)
        return neigh_lr


class AimNet2Mbis(AimNet2Wrapper):
    def __init__(
        self,
        model: Module,
        method_lr: tp.Optional[str] = None,
        cutoff_lr: tp.Optional[float] = None,
        dsf_alpha: tp.Optional[float] = None,
        output_hartree: bool = True,
        strategy: str = "pyaev",
    ) -> None:
        super().__init__(model, method_lr, cutoff_lr, dsf_alpha, output_hartree)
        asm = Assembler(cls=ANIq, periodic_table_index=True)
        asm.set_symbols(SYMBOLS_2X)
        asm.set_global_cutoff_fn("cosine")
        asm.set_aev_computer(radial="ani2x", angular="ani2x", strategy=strategy)
        asm.set_atomic_networks(ctor="ani2x")
        asm.set_atomic_networks(
            cls=_ZeroANINetworks,
            ctor="default",
            kwargs={"bias": False, "activation": "gelu"},
        )
        asm.set_charge_networks(
            cls=_ANINetworksDiscardFirstScalar,
            ctor="ani2x",
            kwargs={"out_dim": 2, "bias": False, "activation": "gelu"},
            normalizer=ChargeNormalizer.from_electronegativity_and_hardness(
                asm.symbols, scale_weights_by_charges_squared=True
            ),
        )
        asm.set_neighborlist("adaptive")
        asm.set_zeros_as_self_energies()  # Dummy energies
        model = tp.cast(ANIq, asm.assemble(8))

        ani2x_state_dict = _fetch_state_dict("ani2x_state_dict.pt")
        aev_state_dict = {
            k.replace("aev_computer.", ""): v
            for k, v in ani2x_state_dict.items()
            if k.startswith("aev_computer")
        }
        charge_state_dict = _fetch_state_dict("charge_nn_state_dict.pt", private=True)
        model.potentials["nnp"].aev_computer.load_state_dict(aev_state_dict)
        model.potentials["nnp"].charge_networks.load_state_dict(charge_state_dict)
        self._charge_model = model

    def forward(
        self,
        species_coords: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        charge: int = 0,
        atomic: bool = False,
        ensemble_values: bool = False,
        _molecule_idxs: tp.Optional[Tensor] = None,
    ) -> SpeciesEnergiesAtomicCharges:
        species, coords = species_coords
        neighbors_lr = self._calc_neighbors(species, coords, cell, pbc, _molecule_idxs)
        species, energies, _ = self._run_aimnet_forward(
            species,
            coords,
            neighbors_lr,
            cell,
            pbc,
            charge,
            atomic,
            ensemble_values,
        )

        elem_idxs = self._charge_model.species_converter(
            species, nop=False,
        )
        # Run charge model only
        atomic_charges = self._charge_model.compute_from_neighbors(
            elem_idxs,
            coords,
            neighbors_lr,
            charge,
            atomic,
            ensemble_values,
        ).scalars
        assert atomic_charges is not None  # mypy
        return SpeciesEnergiesAtomicCharges(species, energies, atomic_charges)


# Utility function useful for debugging and comparing with the input of AimNet
def sort_model_inputs(inputs: tp.Dict[str, Tensor]) -> tp.Dict[str, Tensor]:
    if "shifts" not in inputs:
        return inputs
    for suffix in ("", "_lr"):
        neigh = inputs[f"nbmat{suffix}"]
        mask = inputs[f"nb_pad_mask{suffix}"]
        shifts = inputs[f"shifts{suffix}"]

        sort_result = torch.sort(neigh, dim=-1, stable=True)
        sort_idxs = sort_result.indices
        sort_shift_idxs = sort_idxs.unsqueeze(-1).expand(-1, -1, 3)

        inputs[f"nbmat{suffix}"] = sort_result.values
        inputs[f"nb_pad_mask{suffix}"] = torch.gather(mask, dim=1, index=sort_idxs)
        inputs[f"shifts{suffix}"] = torch.gather(shifts, dim=1, index=sort_shift_idxs)
    return dict(sorted(inputs.items()))


if __name__ == "__main__":
    # TODO: Don't run tests here, it is dirty
    device = "cpu"
    atomic_nums = torch.tensor([[6, 1, 1, 1, 1]], device=device)
    coords = torch.tensor(
        [
            [
                [0.03192167, 0.00638559, 0.01301679],
                [-0.83140486, 0.39370209, -0.26395324],
                [-0.66518241, -0.84461308, 0.20759389],
                [0.45554739, 0.54289633, 0.81170881],
                [0.66091919, -0.16799635, -0.91037834],
            ]
        ],
        device=device,
    )
    expect = {
        "b973c": {
            "atomic_charges": torch.tensor(
                [[-0.41535044, 0.13177463, 0.09962541, 0.10014354, 0.08380468]]
            ),
            "energies": torch.tensor([-40.45836639]),
        },
        "wb97m": {
            "atomic_charges": torch.tensor(
                [[-0.35392734, 0.10948670, 0.08979052, 0.08478487, 0.06986529]]
            ),
            "energies": torch.tensor([-40.49915695]),
        },
    }
    m = make_molec(100, pbc=True, seed=1234, cell_size=30.0, device=device)
    cut = 10.0
    atoms = ase.Atoms(
        positions=m.coords.cpu().numpy().squeeze(0),
        numbers=m.atomic_nums.cpu().numpy().squeeze(0),
        cell=m.cell.cpu().numpy(),
        pbc=m.pbc.cpu().numpy(),
    )
    expect_pbc = {
        "b973c": {
            "ewald": torch.tensor([-4071.60068834]),
            "dsf": torch.tensor([-4071.42818858]),
        },
        "wb97m": {
            "ewald": torch.tensor([-4074.42393669]),
            "dsf": torch.tensor([-4074.20682727]),
        },
    }
    for lot in ("b973c", "wb97m"):
        fname = f"aimnet2_{lot}_0.raw.pt"
        path = (Path(__file__).parent / fname).resolve()
        for kwargs in (
            AimNetArgs("simple"),
            AimNetArgs("ewald", 10.0),  # Amber cutoff must be 10.0
            AimNetArgs("dsf", 10.0, 0.2),  # Amber cutoff must be 10.0
        ):
            model = AimNet2Wrapper.from_jit_file(path, device=device, **asdict(kwargs))
            model_mbis = AimNet2Mbis.from_jit_file(
                path, device=device, **asdict(kwargs)
            )
            if kwargs.method_lr == "simple":
                out = model((atomic_nums, coords))
                out_mbis = model_mbis((atomic_nums, coords))
                # Sanity check, no-pbc (cell is *required* if running with ewald or dsf)
                _expect = expect[lot]
                torch.testing.assert_close(
                    out.energies.cpu().float(), _expect["energies"]
                )
                torch.testing.assert_close(
                    out_mbis.energies.cpu().float(), _expect["energies"]
                )
                torch.testing.assert_close(
                    out.atomic_charges.cpu().float(), _expect["atomic_charges"]
                )
                # MBIS charges are different from AimNet2 charges (hirshfeld)
                assert (out_mbis.atomic_charges != out.atomic_charges).all()
            else:
                # Sanity check, pbc
                out = model((m.atomic_nums, m.coords), m.cell, m.pbc)
                out_mbis = model_mbis((m.atomic_nums, m.coords), m.cell, m.pbc)

                # This doesn't really work correctly in cpu, since the AimNet cpu
                # code is buggy and slow
                if torch.cuda.is_available() and AIMNET2CALC_AVAIL:
                    calc = AIMNet2ASE(f"aimnet2/aimnet2_{lot}_0")
                    calc.base_calc.model.to("cuda")
                    calc.base_calc.device = "cuda"
                    calc.base_calc.set_lrcoulomb_method(
                        method=kwargs.method_lr,
                        cutoff=kwargs.cutoff_lr,
                        dsf_alpha=kwargs.dsf_alpha,
                    )
                    calc.set_atoms(atoms)
                    calc.calculate(properties=["energy"])
                    results = calc.results
                    # Compare directly with AimNet2 results
                    torch.testing.assert_close(
                        torch.from_numpy(results["energy"] / HARTREE_TO_EV).float(),
                        out.energies.cpu().float(),
                    )
                    torch.testing.assert_close(
                        torch.from_numpy(results["energy"] / HARTREE_TO_EV).float(),
                        out_mbis.energies.cpu().float(),
                    )
                # There are slight differences in E due to difference in neighborlist
                # order, which displaces different atoms and generate deltas of the
                # order of f32 precision
                torch.testing.assert_close(
                    expect_pbc[lot][kwargs.method_lr], out.energies.cpu().float()
                )
                torch.testing.assert_close(
                    expect_pbc[lot][kwargs.method_lr], out_mbis.energies.cpu().float()
                )
                # MBIS charges are different from AimNet2 charges (hirshfeld)
                assert (out_mbis.atomic_charges != out.atomic_charges).all()
            out_dir = Path(__file__).parent / "aimnet2"
            out_dir.mkdir(exist_ok=True)

            torch.jit.save(
                torch.jit.script(model), out_dir / f"aimnet2-{lot}-{kwargs.suffix}.pt"
            )
            torch.jit.save(
                torch.jit.script(model_mbis),
                out_dir / f"aimnet2-{lot}-mbis-{kwargs.suffix}.pt",
            )
