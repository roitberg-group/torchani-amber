import torch
from torch import Tensor
from typing import Optional, Tuple
from torch_cluster import radius_graph
import numba

try:
    # optionaly use numba cuda
    import numba.cuda

    _numba_cuda_available = True
except ImportError:
    _numba_cuda_available = False
import numpy as np


def _sparse_nb_to_dense_half(idx: torch.Tensor, natom: int, max_nb: int = 256):
    # Make edges symmetric
    idx_sym = torch.cat([idx, idx[:, [1, 0]]], dim=0)  # shape [2E, 2]
    i, j = idx_sym[:, 0], idx_sym[:, 1]  # i -> j

    # Determine insertion positions via scatter_add
    insert_idx = torch.zeros(natom, dtype=torch.long)
    pos = torch.zeros_like(i)
    insert_idx = torch.cumsum(torch.bincount(i, minlength=natom), dim=0)
    insert_idx = torch.cat([torch.zeros(1, dtype=torch.long), insert_idx[:-1]])
    pos = torch.zeros_like(i)

    # Vectorized with torch.arange and scatter
    idx_sorted, sort_idx = torch.sort(i)
    _, idx_pos = torch.sort(sort_idx)
    pos_unsorted = (
        torch.arange(i.size(0), device=i.device)
        - torch.cumsum(torch.bincount(idx_sorted, minlength=natom)[idx_sorted], dim=0)
        + 1
    )
    pos = pos_unsorted[idx_pos]

    # Mask to exclude neighbors beyond max_nb
    mask = pos < max_nb
    i_masked = i[mask]
    j_masked = j[mask]
    pos_masked = pos[mask]

    # Initialize dense neighbor tensor
    dense_nb = torch.full((natom + 1, max_nb), natom, dtype=torch.long)
    dense_nb[i_masked, pos_masked] = j_masked
    return dense_nb


@numba.njit(cache=True)
def sparse_nb_to_dense_half(idx, natom, max_nb):
    dense_nb = np.full((natom + 1, max_nb), natom, dtype=np.int32)
    last_idx = np.zeros((natom,), dtype=np.int32)
    for k in range(idx.shape[0]):
        i, j = idx[k]
        il, jl = last_idx[i], last_idx[j]
        dense_nb[i, il] = j
        dense_nb[j, jl] = i
        last_idx[i] += 1
        last_idx[j] += 1
    return dense_nb


def nblist_torch_cluster(
    coord: Tensor, cutoff: float, mol_idx: Optional[Tensor] = None, max_nb: int = 256
):
    device = coord.device
    assert coord.ndim == 2, "Expected 2D tensor for coord, got {coord.ndim}D"
    assert coord.shape[0] < 2147483646, "Too many atoms, max supported is 2147483646"
    max_num_neighbors = max_nb
    while True:
        sparse_nb = radius_graph(
            coord, batch=mol_idx, r=cutoff, max_num_neighbors=max_nb
        ).to(torch.int32)
        nnb = torch.unique(sparse_nb[0], return_counts=True)[1]
        if nnb.numel() == 0:
            break
        max_num_neighbors = nnb.max().item()
        if max_num_neighbors < max_nb:
            break
        max_nb *= 2
    sparse_nb_half = sparse_nb[:, sparse_nb[0] > sparse_nb[1]]

    dense_nb = sparse_nb_to_dense_half(
        sparse_nb_half.mT.cpu().numpy(), coord.shape[0], max_num_neighbors
    )
    dense_nb = torch.as_tensor(dense_nb, device=device)
    return dense_nb


# Neighbors(indices=tensor([[1, 7, 7, 8, 2, 3],
# [7, 0, 2, 3, 0, 2]]), distances=tensor([3.9878, 1.5962, 3.3706, 3.8505, 1.9150, 4.9966]), diff_vectors=tensor([[-3.0223,  2.2847,  1.2442],

# sparse tensor([[2, 7, 7, 0, 3, 7, 2, 8, 0, 1, 2, 3],
# [0, 0, 1, 2, 2, 2, 3, 3, 7, 7, 7, 8]], dtype=torch.int32)


# dense neighbor matrix kernels
@numba.njit(cache=True, parallel=True)
def _cpu_dense_nb_mat_sft(conn_matrix):
    N, S = conn_matrix.shape[:2]
    # figure out max number of neighbors
    _s_flat_conn_matrix = conn_matrix.reshape(N, -1)
    maxnb = np.max(np.sum(_s_flat_conn_matrix, axis=-1))
    M = maxnb
    # atom idx matrix
    mat_idxj = np.full((N + 1, M), N, dtype=np.int_)
    # padding matrix
    mat_pad = np.ones((N + 1, M), dtype=np.bool_)
    # shitfs matrix ("S" for "shifts")
    mat_S_idx = np.zeros((N + 1, M), dtype=np.int_)
    for _n in numba.prange(N):
        _i = 0
        for _s in range(S):
            for _m in range(N):
                if conn_matrix[_n, _s, _m] == True:
                    mat_idxj[_n, _i] = _m
                    mat_pad[_n, _i] = False
                    mat_S_idx[_n, _i] = _s
                    _i += 1
    return mat_idxj, mat_pad, mat_S_idx


if _numba_cuda_available:

    @numba.cuda.jit(cache=True)
    def _cuda_dense_nb_mat_sft(conn_matrix, mat_idxj, mat_pad, mat_S_idx):
        i = numba.cuda.grid(1)
        if i < conn_matrix.shape[0]:
            k = 0
            for s in range(conn_matrix.shape[1]):
                for j in range(conn_matrix.shape[2]):
                    if conn_matrix[i, s, j] > 0:
                        mat_idxj[i, k] = j
                        mat_pad[i, k] = 0
                        mat_S_idx[i, k] = s
                        k += 1


def nblists_torch_pbc(
    coord: Tensor, cell: Tensor, cutoff: float
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute dense neighbor lists for periodic boundary conditions case.
    Coordinates must be in cartesian coordinates and be within the unit cell.
    Single crystal only, no support for batched coord or multiple unit cells.
    """
    assert coord.ndim == 2, "Expected 2D tensor for coord, got {coord.ndim}D"
    # non-PBC version
    device = coord.device

    reciprocal_cell = cell.inverse().t()
    inv_distances = reciprocal_cell.norm(2, -1)
    shifts = _calc_shifts(inv_distances, cutoff)
    d = torch.cdist(
        coord.unsqueeze(0), coord.unsqueeze(0) + (shifts @ cell).unsqueeze(1)
    )
    conn_mat = ((d < cutoff) & (d > 0.1)).transpose(0, 1).contiguous()
    if device.type == "cuda" and _numba_cuda_available:
        _fn = _nblist_pbc_cuda
    else:
        _fn = _nblist_pbc_cpu
    mat_idxj, mat_pad, mat_S = _fn(conn_mat, shifts)
    return mat_idxj, mat_pad, mat_S


def _calc_shifts(inv_distances, cutoff):
    num_repeats = torch.ceil(cutoff * inv_distances).to(torch.long)
    dc = [
        torch.arange(-num_repeats[i], num_repeats[i] + 1, device=inv_distances.device)
        for i in range(len(num_repeats))
    ]
    shifts = torch.cartesian_prod(*dc).to(torch.float)
    return shifts


def _nblist_pbc_cuda(conn_mat, shifts):
    N = conn_mat.shape[0]
    M = conn_mat.view(N, -1).sum(-1).max()
    threadsperblock = 32
    blockspergrid = (N + (threadsperblock - 1)) // threadsperblock
    idx_j = torch.full((N + 1, M), N, dtype=torch.int64, device=conn_mat.device)
    mat_pad = torch.ones((N + 1, M), dtype=torch.int8, device=conn_mat.device)
    S_idx = torch.zeros((N + 1, M), dtype=torch.int64, device=conn_mat.device)
    conn_mat = conn_mat.to(torch.int8)
    _conn_mat = numba.cuda.as_cuda_array(conn_mat)
    _idx_j = numba.cuda.as_cuda_array(idx_j)
    _mat_pad = numba.cuda.as_cuda_array(mat_pad)
    _S_idx = numba.cuda.as_cuda_array(S_idx)
    _cuda_dense_nb_mat_sft[blockspergrid, threadsperblock](
        _conn_mat, _idx_j, _mat_pad, _S_idx
    )
    mat_pad = mat_pad.to(torch.bool)
    return idx_j, mat_pad, shifts[S_idx]


def _nblist_pbc_cpu(conn_mat, shifts):
    conn_mat = conn_mat.cpu().numpy()
    mat_idxj, mat_pad, mat_S_idx = _cpu_dense_nb_mat_sft(conn_mat)
    mat_idxj = torch.from_numpy(mat_idxj).cpu()
    mat_pad = torch.from_numpy(mat_pad).cpu()
    mat_S_idx = torch.from_numpy(mat_S_idx).cpu()
    mat_S = shifts[mat_S_idx]
    return mat_idxj, mat_pad, mat_S


from torchani.neighbors import CellList, AllPairs, reconstruct_shifts
from torchani._testing import make_molec

# def nblist_torch_cluster(
# coord: Tensor, cutoff: float, mol_idx: Optional[Tensor] = None, max_nb: int = 256
# ):
# pass
atoms_num = 1000
# seed 2343 fails
for seed in (1234, 2343, 3434, 23432, 2342, 233):
    print(seed)
    m = make_molec(atoms_num, cell_size=30.0, pbc=True, seed=seed)
    cut = 10.0
    clist = CellList()
    neigh = clist(
        cutoff=cut, species=m.atomic_nums, coords=m.coords, pbc=m.pbc, cell=m.cell
    )
    idx = neigh.indices
    num_neighbors = torch.bincount(idx.view(-1), minlength=atoms_num + 1)
    max_neighbors = int(num_neighbors.max().item())
    mask = (
        num_neighbors.unsqueeze(1)
        > torch.arange(max_neighbors, device=idx.device, dtype=idx.dtype)
    ).to(torch.bool)
    shift_idxs = torch.round(
        reconstruct_shifts(m.coords, neigh) @ torch.inverse(m.cell)
    ).to(torch.long)

    full_shiftidxs = -torch.cat((shift_idxs, -shift_idxs), dim=0)
    full_idx = torch.cat((idx, idx.flipud()), dim=1)
    sort_idxs = torch.sort(full_idx[0]).indices
    # This +1 is not really useful but is added for compatibility
    dense_nb = idx.new_full((atoms_num + 1, max_neighbors), fill_value=atoms_num)
    dense_nb.masked_scatter_(mask, full_idx[1, sort_idxs])

    # Now they need to be masked-scattered into the array
    dense_shifts = idx.new_full((atoms_num + 1, max_neighbors, 3), fill_value=-1)
    dense_shifts.masked_scatter_(mask.unsqueeze(-1), full_shiftidxs[sort_idxs])
    dense_shifts = dense_shifts.to(torch.float)

    sort_result = torch.sort(dense_nb, dim=-1, stable=True)
    dense_nb = sort_result.values
    idxs = sort_result.indices.unsqueeze(-1).expand(-1, -1, 3)
    dense_shifts = torch.gather(dense_shifts, dim=1, index=idxs)

    # I believe the idxs have to be flipped with a - sign, and
    out = nblists_torch_pbc(m.coords.squeeze(0), m.cell, cut)
    target_nb = out[0]
    target_mask = out[1]
    target_shifts = out[2]

    sort_result2 = torch.sort(target_nb, dim=-1, stable=True)
    target_nb = sort_result2.values
    idxs2 = sort_result2.indices.unsqueeze(-1).expand(-1, -1, 3)
    target_shifts = torch.gather(target_shifts, dim=1, index=idxs2)

    # tensor([[7, 2, 0],
    #        [7, 3, 0]])
    if not (target_shifts == dense_shifts).all():
        differ = (target_shifts != dense_shifts).nonzero()
        for triple in differ.unbind():
            if target_nb[triple[0], triple[1]] != target_nb[triple[0], triple[1]]:
                breakpoint()

# I believe the padding value is -1 -1 -1
# The shifts matrix has shape (atoms + 1, max_neighbors, 3) and has the 1 1 1, -1 -1 -1
# etc shifts. It holds, for each neighbor, the associated shift vector
