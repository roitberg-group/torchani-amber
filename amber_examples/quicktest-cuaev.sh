#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Run a quick cuda pmemd test with torchani



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

echo "############# Running Quick Test #############"
echo "\n\nsander CUDA"
time sander\
    -O\
    -i ./inputs/quicktest-cuaev.in\
    -p tripeptide.prmtop\
    -c tripeptide_min.ncrst\
    -o outputs/quicktest-cuaev.out\
    -r tmp\
    -inf tmp\
    -x outputs/quicktest-cuaev.nc
rm tmp
cd - || exit 1
