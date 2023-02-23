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

echo "############# Running CPU dynamics #############"
echo "############# Single precision #############"
echo "\n\npmemd CPU"
time pmemd -O -i ./inputs/dynamics_cpu_single_ani1x.in \
         -p tripeptide.prmtop \
         -c tripeptide_min.ncrst \
         -o outputs/dynamics_pmemd_cpu_single_ani1x.out \
         -r tmp \
         -inf tmp \
         -x outputs/dynamics_pmemd_cpu_single_ani1x.nc
echo "\n\nsander CPU"
time sander -O -i ./inputs/dynamics_cpu_single_ani1x.in \
         -p tripeptide.prmtop \
         -c tripeptide_min.ncrst \
         -o outputs/dynamics_sander_cpu_single_ani1x.out \
         -r tmp \
         -inf tmp \
         -x outputs/dynamics_sander_cpu_single_ani1x.nc
echo "############# Double precision #############"
echo "\n\npmemd CPU"
time pmemd -O -i ./inputs/dynamics_cpu_double_ani1x.in \
         -p tripeptide.prmtop \
         -c tripeptide_min.ncrst \
         -o outputs/dynamics_pmemd_cpu_double_ani1x.out \
         -r tmp \
         -inf tmp \
         -x outputs/dynamics_pmemd_cpu_double_ani1x.nc
echo "\n\nsander CPU"
time sander -O -i ./inputs/dynamics_cpu_double_ani1x.in \
         -p tripeptide.prmtop \
         -c tripeptide_min.ncrst \
         -o outputs/dynamics_sander_cpu_double_ani1x.out \
         -r tmp \
         -inf tmp \
         -x outputs/dynamics_sander_cpu_double_ani1x.nc

if $(hash nvidia-smi); then
    echo "############# Running CUDA dynamics #############"
    echo "############# Single precision #############"
    echo
    echo
    echo "pmemd CUDA"
    time pmemd -O -i ./inputs/dynamics_cuda_single_ani1x.in \
             -p tripeptide.prmtop \
             -c tripeptide_min.ncrst \
             -o outputs/dynamics_pmemd_cuda_single_ani1x.out \
             -r tmp \
             -inf tmp \
             -x outputs/dynamics_pmemd_cuda_single_ani1x.nc
    echo
    echo
    echo "sander CUDA"
    time sander -O -i ./inputs/dynamics_cuda_single_ani1x.in \
             -p tripeptide.prmtop \
             -c tripeptide_min.ncrst \
             -o outputs/dynamics_sander_cuda_single_ani1x.out \
             -r tmp \
             -inf tmp \
             -x outputs/dynamics_sander_cuda_single_ani1x.nc
    echo "############# Double precision #############"
    echo
    echo
    echo "pmemd CUDA"
    time pmemd -O -i ./inputs/dynamics_cuda_double_ani1x.in \
             -p tripeptide.prmtop \
             -c tripeptide_min.ncrst \
             -o outputs/dynamics_pmemd_cuda_double_ani1x.out \
             -r tmp \
             -inf tmp \
             -x outputs/dynamics_pmemd_cuda_double_ani1x.nc
    echo
    echo
    echo "sander CUDA"
    time sander -O -i ./inputs/dynamics_cuda_double_ani1x.in \
             -p tripeptide.prmtop \
             -c tripeptide_min.ncrst \
             -o outputs/dynamics_sander_cuda_double_ani1x.out \
             -r tmp \
             -inf tmp \
             -x outputs/dynamics_sander_cuda_double_ani1x.nc
    echo "############# Neighborlist Large #############"
    echo
    echo
    echo "pmemd CUDA"
    time pmemd -O -i ./inputs/dynamics_cuda_double_nl.in \
             -p large_water.parm7 \
             -c large_water.rst7 \
             -o outputs/dynamics_pmemd_cuda_double_nl_large.out \
             -r tmp \
             -inf tmp \
             -x outputs/dynamics_pmemd_cuda_double_nl_large.nc
    echo
    echo
    echo "sander CUDA"
    time sander -O -i ./inputs/dynamics_cuda_double_nl.in \
             -p large_water.parm7 \
             -c large_water.rst7 \
             -o outputs/dynamics_sander_cuda_double_nl.out \
             -r tmp \
             -inf tmp \
             -x outputs/dynamics_sander_cuda_double_nl.nc
    echo "############# Neighborlist #############"
    echo
    echo
    echo "pmemd CUDA"
    time pmemd -O -i ./inputs/dynamics_cuda_double_nl.in \
             -p tripeptide.prmtop \
             -c tripeptide_min.ncrst \
             -o outputs/dynamics_pmemd_cuda_double_nl.out \
             -r tmp \
             -inf tmp \
             -x outputs/dynamics_pmemd_cuda_double_nl.nc
    echo
    echo
    echo "sander CUDA"
    time sander -O -i ./inputs/dynamics_cuda_double_nl.in \
             -p tripeptide.prmtop \
             -c tripeptide_min.ncrst \
             -o outputs/dynamics_sander_cuda_double_nl.out \
             -r tmp \
             -inf tmp \
             -x outputs/dynamics_sander_cuda_double_nl.nc
             else
    echo "WARNING: No CUDA devices detected, only running CPU dynamics"
    echo "To run CUDA tests in the future execute './amber_inputs/run.sh'"
fi

rm tmp
cd - || exit 1
