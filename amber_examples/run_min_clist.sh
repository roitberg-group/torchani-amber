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
# -r .ncrst coordinates restart file (non formated) can be read as -c
# -inf Information about the simulation
# -x trajectory output file (non formatted)
cd "${DIR}" || exit 1
if [ ! -d "outputs" ]; then
    mkdir outputs
fi

echo "############# Neighborlist Min #############"
echo "\n\npmemd CUDA"
time pmemd -O -i ./inputs/min.in \
         -p large_water.parm7 \
         -c large_water2.ncrst \
         -o outputs/large_water_min_nl.out \
         -r large_water3.ncrst \
         -inf tmp \
         -x outputs/large_water_min_nl.nc
