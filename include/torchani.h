#ifndef TORCHANI_AMBER_INTERFACE_SRC_TORCHANI_H_
#define TORCHANI_AMBER_INTERFACE_SRC_TORCHANI_H_


extern "C" void torchani_init_atom_types_(
    int atomic_numbers[],
    int* num_atoms_raw,
    int* use_cuda_device_raw,
    int* device_index_raw,
    int* use_double_precision_raw,
    int* model_type_raw,
    int* model_index_raw,
    int* use_cell_list_raw,
    int* use_external_neighborlist
);

extern "C" void torchani_energy_force_pbc_(
    double coordinates_raw[][3],
    int* num_atoms_raw,
    double forces[][3],
    double pbc_box[][3],
    double* potential_energy
);

extern "C" void torchani_energy_force_(
    double coordinates_raw[][3],
    int* num_atoms_raw,
    double forces[][3],
    double* potential_energy
);


// accept an external neighborlist and shifts
extern "C" void torchani_energy_force_pbc_external_neighborlist_(
    int* ani_neighborlist[],
    double ani_shifts[][3],
    int* num_neighbors_raw,
    double coordinates_raw[][3],
    int* num_atoms_raw,
    double forces[][3],
    double* potential_energy
);

#endif  // TORCHANI_AMBER_INTERFACE_SRC_TORCHANI_H_
