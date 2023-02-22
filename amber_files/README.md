# Modified amber files

This directory contains amber files with the necessary modifications to
correctly link both pmemd and sander with torchani. Currently amber must be
built using the legacy system (make instead of cmake).

## Necessary steps for linking

Note that it is not necessary to perform the following steps manually, they are
automatically performed by the script `install.sh`, or
`modify_and_link_amber.sh`.

`AMBERHOME` must be set to the main amber directory (where `configure`,
`AmberTools`, `src`, and other files are located)

- `configure2` must be copied into `AMBERHOME/AmberTools/src`
- all the pmemd files must be copied into `AMBERHOME/src/pmemd/src`
- all the sander files must be copied into `AMBERHOME/AmberTools/src/sander`

After this Amber must be configured by doing:

```bash
cd AMBERHOME
./configure -libtorchani -noX11 --skip-python gnu
```

(or whatever flags you want to add). After this `amber.sh` must be sourced:

```bash
cd AMBERHOME
source amber.sh
```

Then `../build/libtorchani.so` has to be copied into `AMBERHOME/lib`
Then pmemd and sander can be directly built without necessarily building all of
Amber by:

```bash
cd AMBERHOME/src/pmemd/src
make
cd AMBERHOME/AmberTools/src/sander
make
```

Alternatively Amber can be built by:

```bash
cd AMBERHOME
make
```
