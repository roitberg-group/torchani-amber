#ifndef TORCHANI_AMBER_INTERFACE_SRC_TORCHANI_H_
#define TORCHANI_AMBER_INTERFACE_SRC_TORCHANI_H_


extern "C" void torchani_init_atom_types_(
    int atomic_numbers[],
    int* num_atoms_raw,
    int* device_index_raw,
    int* torchani_model_index_raw,
    int* network_index_raw,
    int* use_double_precision_raw,
    int* use_cuda_device_raw,
    int* use_torch_cell_list_raw,
    int* use_external_neighborlist_raw
);

extern "C" void torchani_energy_force_(
    double coordinates_raw[][3],
    int* num_atoms_raw,
    /* outputs */
    double forces[][3],
    double* potential_energy
);


extern "C" void torchani_energy_force_qbc_(
    double coordinates_raw[][3],
    int* num_atoms_raw,
    /* outputs */
    double forces[][3],
    double* potential_energy,
    double* qbc
);


extern "C" void torchani_energy_force_pbc_(
    double coordinates_raw[][3],
    int* num_atoms_raw,
    double forces[][3],
    double pbc_box[][3],
    /* outputs */
    double* potential_energy
);

// accept an external neighborlist and shifts
extern "C" void torchani_energy_force_external_neighborlist_(
    double coordinates_raw[][3],
    int* num_atoms_raw,
    int* num_neighbors_raw,
    int* neighborlist_raw[],
    double shifts_raw[][3],
    /* outputs */
    double forces[][3],
    double* potential_energy
);

#endif  // TORCHANI_AMBER_INTERFACE_SRC_TORCHANI_H_
