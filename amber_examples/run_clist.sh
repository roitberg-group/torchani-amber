#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Run a series of pmemd and sander small trajectories to check that
# the interface works correctly in single and double precision, with and
# without CUDA.

# pmemd variables:
# -O overwrite file if exists
# -i Input file
# -o Output file (human readable)
# -p parm7/prmtop parameters and topology file
# -c rst7 initial coordinates file
# -r .ncrst coordinates restart file (non formated) can be read as -i
# -inf Information about the simulation
# -x trajectory output file (non formatted)
cd "${DIR}" || exit 1
if [ ! -d "outputs" ]; then
    mkdir outputs
fi

echo "############# Neighborlist Large #############"
echo "\n\npmemd CUDA"
time pmemd -O -i ./inputs/dynamics_cuda_double_nl.in \
         -p large_water.parm7 \
         -c large_water3.ncrst \
         -o outputs/dynamics_pmemd_cuda_double_nl_large.out \
         -r tmp \
         -inf tmp \
         -x outputs/dynamics_pmemd_cuda_double_nl_large.nc
echo "\n\nsander CUDA"
time sander -O -i ./inputs/dynamics_cuda_double_nl.in \
         -p large_water.parm7 \
         -c large_water3.ncrst \
         -o outputs/dynamics_sander_cuda_double_nl_large.out \
         -r tmp \
         -inf tmp \
         -x outputs/dynamics_sander_cuda_double_nl_large.nc


echo "############# Neighborlist #############"
echo "\n\npmemd CUDA"
time pmemd -O -i ./inputs/dynamics_cuda_double_nl.in \
         -p tripeptide.prmtop \
         -c tripeptide_min.ncrst \
         -o outputs/dynamics_pmemd_cuda_double_nl.out \
         -r tmp \
         -inf tmp \
         -x outputs/dynamics_pmemd_cuda_double_nl.nc
echo "\n\nsander CUDA"
time sander -O -i ./inputs/dynamics_cuda_double_nl.in \
         -p tripeptide.prmtop \
         -c tripeptide_min.ncrst \
         -o outputs/dynamics_sander_cuda_double_nl.out \
         -r tmp \
         -inf tmp \
         -x outputs/dynamics_sander_cuda_double_nl.nc
rm tmp
cd - || exit 1
