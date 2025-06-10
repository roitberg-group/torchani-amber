from pathlib import Path
import typing as tp

import torch
from torch import Tensor
from torchani.units import HARTREE_TO_KJOULEPERMOL


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
        model,
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
        self._kjpermol_to_hartree = 1 / HARTREE_TO_KJOULEPERMOL
        self._angstrom_to_nm = 0.1

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
        species = species.squeeze(0)
        # Nutmeg models take coords in nm
        coords = coords.squeeze(0) * self._angstrom_to_nm
        types, node_attrs = self._create_atom_features(species, coords)
        energy = self._model(coords, types, node_attrs, cell).view(1)
        return species, energy * self._kjpermol_to_hartree


# TODO: Add large model
for size in ("small", "medium"):
    model = NutmegWrapper(torch.jit.load(f"./nutmeg-{size}.raw.pt"))
    out_dir = Path(__file__).parent / "nutmeg"
    out_dir.mkdir(exist_ok=True)
    torch.jit.save(torch.jit.script(model), out_dir / f"nutmeg-{size}.pt")

# model.set_atomic_charges(
# list(map(float, Path("./ace-ala-nme.charge").read_text().split()))
# )
