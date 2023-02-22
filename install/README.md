```
source leaprc.protein.ff14SB
source leaprc.water.tip3p

unit = sequence { ALA }
solvateBox unit TIP3PBOX 2.5
saveAmberParm unit unit.prmtop box_2p5.rst7

quit
```
