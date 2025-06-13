from pathlib import Path
import typing as tp
import warnings

import torch
from torch import Tensor
from torch.nn import Module
from torchani.units import HARTREE_TO_KJOULEMOL
from torchani.neighbors import (
    reconstruct_shifts,
    AdaptiveList,
    discard_inter_molecule_pairs,
    Neighbors,
)


# Wrapper to run the Nutmeg potentials https://github.com/openmm/nutmeg/
class NutmegWrapper(torch.nn.Module):
    _ZNUM_IDX_MAP = {
        1: 0,
        3: 1,
        5: 2,
        6: 3,
        7: 4,
        8: 5,
        9: 6,
        11: 7,
        12: 8,
        14: 9,
        15: 10,
        16: 11,
        17: 12,
        19: 13,
        20: 14,
        35: 15,
        53: 16,
    }
    _atomic_charges: Tensor
    _types_map: Tensor

    def __init__(
        self,
        model: Module,
        atomic_charges: tp.Sequence[float] = (),
    ) -> None:
        super().__init__()
        self._model = model
        # Atomic charges must be Gasteiger for the Nutmeg models
        _types_map = torch.full((118,), fill_value=-1, dtype=torch.long)
        for k, v in self._ZNUM_IDX_MAP.items():
            _types_map[k] = v
        self.register_buffer("_atomic_charges", torch.tensor(atomic_charges))
        self.register_buffer("_types_map", _types_map)
        self._kjpermol_to_hartree = 1 / HARTREE_TO_KJOULEMOL
        self._angstrom_to_nm = 0.1
        self._cutoff = model.cutoff / self._angstrom_to_nm
        if model.self_interaction:
            raise ValueError("Self-interacting Nutmeg models are not supported")
        param = next(iter(model.parameters()))
        self._nlist = AdaptiveList().to(device=param.device, dtype=param.dtype)

    @torch.jit.export
    def set_extra_features(self, extra_features: tp.List[float]) -> None:
        self._atomic_charges = torch.tensor(
            extra_features,
            dtype=self._atomic_charges.dtype,
            device=self._atomic_charges.device,
        )

    def _create_atom_features(
        self, species: Tensor, coords: Tensor
    ) -> tp.Tuple[Tensor, Tensor]:
        atom_types = self._types_map[species].long()
        one_hot_z = torch.nn.functional.one_hot(atom_types, num_classes=17).to(
            coords.dtype
        )
        atomic_charges = self._atomic_charges.to(
            dtype=coords.dtype, device=coords.device
        )
        return atom_types, torch.cat([one_hot_z, atomic_charges.view(-1, 1)], dim=1)

    def _calc_neighbors(
        self,
        species: Tensor,
        coords: Tensor,
        cell: tp.Optional[Tensor],
        pbc: tp.Optional[Tensor],
        _molecule_idxs: tp.Optional[Tensor],
    ) -> Neighbors:
        # Neighborlist is "complete" (duplicated)
        # edge_attrs is None for these models
        neigh_lr = self._nlist(self._cutoff, species, coords.detach(), cell, pbc)
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

    def forward(
        self,
        species_coords: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        charge: int = 0,
        atomic: bool = False,
        ensemble_values: bool = False,
        _molecule_idxs: tp.Optional[Tensor] = None,
    ) -> tp.Tuple[Tensor, Tensor]:

        species, coords = species_coords
        neigh = self._calc_neighbors(species, coords, cell, pbc, _molecule_idxs)
        full_idxs = torch.cat((neigh.indices, neigh.indices.flipud()), dim=1)
        species = species.squeeze(0)
        coords = coords.squeeze(0)
        if pbc is not None:
            assert cell is not None
            inv_cell = torch.inverse(cell)
            shiftidxs = torch.round(
                reconstruct_shifts(coords.unsqueeze(0), neigh) @ inv_cell
            )
            # TODO: Make sure the -sign is needed
            full_shiftidxs = -torch.cat((shiftidxs, -shiftidxs), dim=0).to(coords.dtype)
        else:
            pbc = species.new_zeros(3, dtype=torch.bool)
            full_shiftidxs = coords.new_zeros((full_idxs.size(1), 3))

        types, node_attrs = self._create_atom_features(species, coords)
        if cell is None:
            cell = torch.eye(3, dtype=coords.dtype, device=coords.device)
        data = {
            "coordinates": coords * self._angstrom_to_nm,
            "edge_index": full_idxs,
            "cell_shift_vector": full_shiftidxs,
            "raw_atomic_numbers": types,
            "node_attrs": node_attrs,
            "num_nodes": species.new_full((1,), fill_value=coords.size(0)),
            "batch": species.new_zeros(coords.size(0)),
            "num_graphs": species.new_ones((1,)),
            "pbc": pbc,
            "cell": cell,
        }
        # Call core model, without neighborlist
        energy = self._model.model(data)["y_graph_scalars"].view(1)
        return species, energy * self._kjpermol_to_hartree


for size in ("small", "medium", "large"):
    model = NutmegWrapper(torch.jit.load(f"./nutmeg-{size}.raw.pt"))
    out_dir = Path(__file__).parent / "nutmeg"
    out_dir.mkdir(exist_ok=True)
    torch.jit.save(torch.jit.script(model), out_dir / f"nutmeg-{size}.pt")
