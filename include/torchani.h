#ifndef TORCHANI_H_
#define TORCHANI_H_


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


extern "C" void torchani_energy_force_atomic_charges_(
    double coordinates_raw[][3],
    int* num_atoms_raw,
    int* charges_type_raw,
    /* outputs */
    double forces[][3],
    double* potential_energy,
    double* atomic_charges
);


extern "C" void torchani_energy_force_atomic_charges_with_derivatives_(
    double coordinates_raw[][3],
    int* num_atoms_raw,
    int* charges_type_raw,
    /* outputs */
    double forces[][3],
    double* potential_energy,
    // Note that atomic_charge_derivatives is a [num_atoms x num_atoms x 3] array
    double* atomic_charge_derivatives,
    // Note that atomic_charges is a [num_atoms] array
    double* atomic_charges
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
#endif  // TORCHANI_H_
