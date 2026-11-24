#pragma once

#include <array>
#include <cstddef>
#include <string_view>

namespace constants {

// Factor used by Amber
constexpr double CODATA08_ANGSTROM_TO_BOHR = 1.8897261328727875;

// The pol_<symbol> options of the ML/MM config are in atomic units
constexpr double POLARIZABILITY_AU_TO_ANG3 = 0.1481847;

// This factor should come straight from torchani.units and be consistent with ASE
constexpr double HARTREE_TO_KCALMOL = 627.5094738898777;
// NOTE: Probably this should be used instead (or nothing)
// double AMBER_HARTREE_TO_KCALMOL = 627.509469433585;

constexpr std::array<std::string_view, 36> mlip_interface_builtin_models{
    "ani1x",
    "ani1x-infer",
    "ani1ccx",
    "ani1ccx-infer",
    "ani2x",
    "ani2x-infer",
    "ani2xr",
    "ani2xr-infer",
    "ani2dr",
    "ani2dr-infer",
    "animbis",
    "animbis-infer",
    "animbis-custom-mlmm",
    "anir2s",
    "anir2s-infer",
    "anir2s_water",
    "anir2s_water-infer",
    "anir2s_chcl3",
    "anir2s_chcl3-infer",
    "anir2s_ch3cn",
    "anir2s_ch3cn-infer",
    "aimnet2-b973c-dsf",
    "aimnet2-b973c-ewald",
    "aimnet2-b973c-nocut",
    "aimnet2-b973c-mbis-dsf",
    "aimnet2-b973c-mbis-ewald",
    "aimnet2-b973c-mbis-nocut",
    "aimnet2-wb97m-dsf",
    "aimnet2-wb97m-ewald",
    "aimnet2-wb97m-nocut",
    "aimnet2-wb97m-mbis-dsf",
    "aimnet2-wb97m-mbis-ewald",
    "aimnet2-wb97m-mbis-nocut",
    "nutmeg-small",
    "nutmeg-medium",
    "nutmeg-large",
};

constexpr std::size_t NUM_ELEMENTS = 118;

// Indexed by atomic number - 1
constexpr std::array<std::string_view, NUM_ELEMENTS> ELEMENT_SYMBOLS{
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
};

}  // namespace constants
