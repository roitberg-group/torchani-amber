#include <torchani.h>

#include <dlfcn.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <ATen/Context.h>
#include <c10/util/Logging.h>
#include <torch/all.h>
#include <torch/csrc/autograd/autograd.h>  // For forces

#include "tensor_utils.h"
#include "config.h"
#include "constants.h"
#include "electro.h"
#include "io.h"
#include "mlip.h"
#include "timing.h"

// Globals:
namespace {
// Default device is CPU, and default kind is Float32
config::GlobalConfig global_config{};
std::unique_ptr<mlip::MlipModel> model;
// Tensors stored through the C API, and the graph kept between
// Cleared by mlip_initialize.
// Names starting with "__" are reserved for internal use
tensor_utils::TensorContainer tensor_container;
// Set by mlip_update_md_step. The write_* outputs are written once per step, if it is a
// multiple of write_interval
int current_md_step = 0;
std::optional<int> last_written_md_step;  // Reset by mlip_initialize

void populateEmbeddingForces(
    torch::Tensor ml_forces,
    torch::Tensor mm_forces,
    int ml_atoms_num,
    int mm_atoms_num,
    double ml_forces_buf[][3],
    double mm_forces_buf[][3]
) {
    ml_forces = ml_forces * constants::HARTREE_TO_KCALMOL;
    mm_forces = mm_forces * constants::HARTREE_TO_KCALMOL;
    // Accessor is used to access tensor elements, but dimensionality has
    // to be known at compile time. Note that this is a CPU accessor.
    // Packed accessors can't be used to access CUDA tensors outside CUDA kernels,
    // AFIK cuda tensors have to be converted to cpu before access is allowed
    // This cast does nothing if output is already of the required type and
    // device, so there should be no overhead.
    ml_forces = ml_forces.to(torch::kCPU, torch::kDouble);
    auto force_a = ml_forces.accessor<double, 3>();
    for (int atom = 0; atom != ml_atoms_num; ++atom) {
        for (int c = 0; c != 3; ++c) {
            ml_forces_buf[atom][c] = force_a[0][atom][c];
        }
    }
    mm_forces = mm_forces.to(torch::kCPU, torch::kDouble);
    auto qforce_a = mm_forces.accessor<double, 3>();
    for (int q = 0; q != mm_atoms_num; ++q) {
        for (int c = 0; c != 3; ++c) {
            mm_forces_buf[q][c] = qforce_a[0][q][c];
        }
    }
}

void populateForces(torch::Tensor forces, double forces_buf[][3], int num_atoms) {
    forces = forces * constants::HARTREE_TO_KCALMOL;
    // Accessor is used to access tensor elements, but dimensionality has
    // to be known at compile time. Note that this is a CPU accessor.
    // Packed accessors can't be used to access CUDA tensors outside CUDA kernels,
    // AFIK cuda tensors have to be converted to cpu before access is allowed
    // This cast does nothing if output is already of the required type and
    // device, so there should be no overhead.
    forces = forces.to(torch::kCPU, torch::kDouble);
    if (global_config.useNoEvalAtoms()) {
        auto do_eval_idxs = tensor_container.get("__do_eval_atoms_idxs").to(torch::kCPU);
        auto do_eval_idxs_a = do_eval_idxs.accessor<long, 1>();
        auto forces_a = forces.accessor<double, 3>();
        for (int atom = 0; atom != num_atoms; ++atom) {
            for (int c = 0; c != 3; ++c) {
                forces_buf[atom][c] = 0.0;
            }
        }
        for (long i = 0; i != do_eval_idxs.size(0); ++i) {
            for (int c = 0; c != 3; ++c) {
                forces_buf[do_eval_idxs_a[i]][c] = forces_a[0][i][c];
            }
        }
    } else {
        auto forces_a = forces.accessor<double, 3>();
        for (int atom = 0; atom != num_atoms; ++atom) {
            for (int c = 0; c != 3; ++c) {
                forces_buf[atom][c] = forces_a[0][atom][c];
            }
        }
    }
}

void populateAtomicCharges(
    torch::Tensor& atomic_charges_tensor, double* atomic_charges, int num_atoms
) {
    atomic_charges_tensor = atomic_charges_tensor.to(torch::kCPU, torch::kDouble);
    auto atomic_charges_accessor = atomic_charges_tensor.accessor<double, 2>();

    for (int atom = 0; atom != num_atoms; ++atom) {
        atomic_charges[atom] = atomic_charges_accessor[0][atom];
    }
}

void populateEnergyScalar(torch::Tensor& output, double* energy_scalar) {
    // data is copied here, otherwise errors ocurr, since the data dies as soon
    // as the function exits.
    // This cast does nothing if output is already of the required type and
    // device, so there should be no overhead
    output = output.to(torch::kCPU, torch::kDouble);
    *energy_scalar = (*output.data_ptr<double>()) * constants::HARTREE_TO_KCALMOL;
}

void exitIfNoEvalAtoms(
    const config::GlobalConfig& config,
    std::source_location loc = std::source_location::current()
) {
    if (config.useNoEvalAtoms()) {
        std::cerr << "ERROR (libtorchani)\n"
                  << loc.function_name() << " does not support"
                  << " atoms excluded from the evaluation" << std::endl;
        std::exit(2);
    }
}

void exitIfNoMlmmConfig(
    const config::GlobalConfig& config,
    std::source_location loc = std::source_location::current()
) {
    if (not config.hasMlmmConfig()) {
        std::cerr << "ERROR (libtorchani)\n"
                  << loc.function_name() << " requires the ML/MM config, call"
                  << " mlip_initialize with read_mlmm_config = true first" << std::endl;
        std::exit(2);
    }
}

void exitIfNotInitialized(
    const config::GlobalConfig& config,
    std::source_location loc = std::source_location::current()
) {
    if (not config.mlipInterfaceIsInit()) {
        std::cerr << "ERROR (libtorchani)\n"
                  << loc.function_name() << " was called before mlip_initialize" << std::endl;
        std::exit(2);
    }
}

// Namelist keys are lowercase
std::string lowercase(std::string str) {
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
    return str;
}



// Atomic numbers of the ML atoms, on the CPU
std::vector<long> mlAtomicNumbers() {
    torch::Tensor atomic_numbers = tensor_container.get("__atomic_numbers");
    torch::Tensor atomic_nums = atomic_numbers.view({-1}).to(torch::kCPU);
    return std::vector<long>(
        atomic_nums.data_ptr<long>(), atomic_nums.data_ptr<long>() + atomic_nums.numel()
    );
}

// Polarizabilities of the ML atoms (AU), from the ML/MM config. Exits if the
// ML/MM config was not read, or if an element has no polarizability
torch::Tensor mlAtomicPolarizabilities() {
    std::vector<double> alphas;
    for (long atomic_num : mlAtomicNumbers()) {
        double alpha = std::numeric_limits<double>::quiet_NaN();
        try {
            alpha = global_config.getPolarizability(atomic_num);
        } catch (const std::exception& e) {
            std::cerr << "ERROR (libtorchani)\n" << e.what() << std::endl;
            std::exit(2);
        }
        alphas.push_back(alpha);
    }
    return global_config.tensorToTorchaniDtypeAndDevice(
        torch::tensor(alphas, torch::TensorOptions().dtype(torch::kDouble))
    );
}

// Factors kz of the ML atoms (AU), such that their polarizabilities are kz times their
// atomic volumes, from the ML/MM config. Exits if the ML/MM config was not read, or if
// an element has no kz
torch::Tensor mlVolumeKzs() {
    std::vector<double> kzs;
    for (long atomic_num : mlAtomicNumbers()) {
        double kz = std::numeric_limits<double>::quiet_NaN();
        try {
            kz = global_config.getVolumeKz(atomic_num);
        } catch (const std::exception& e) {
            std::cerr << "ERROR (libtorchani)\n" << e.what() << std::endl;
            std::exit(2);
        }
        kzs.push_back(kz);
    }
    return global_config.tensorToTorchaniDtypeAndDevice(
        torch::tensor(kzs, torch::TensorOptions().dtype(torch::kDouble))
    );
}

void printPolarizabilitiesInUse() {
    std::set<long> atomic_nums;
    for (long atomic_num : mlAtomicNumbers()) {
        atomic_nums.insert(atomic_num);
    }
    bool use_mlip_atomic_volumes = global_config.getBool("use_mlip_atomic_volumes");
    if (use_mlip_atomic_volumes) {
        std::cout << "ML/MM: The following kz factors (alpha = kz * volume) are in use:\n"
                  << "Element Symbol, kz (AU)\n";
    } else {
        std::cout << "ML/MM: The following atomic polarizabilities (alpha) are in use:\n"
                  << "Element Symbol, Alpha (AU)\n";
    }
    for (long atomic_num : atomic_nums) {
        double value = 0.0;
        try {
            if (use_mlip_atomic_volumes) {
                value = global_config.getVolumeKz(atomic_num);
            } else {
                value = global_config.getPolarizability(atomic_num);
            }
        } catch (const std::exception& e) {
            std::cerr << "ERROR (libtorchani)\n" << e.what() << std::endl;
            std::exit(2);
        }
        std::cout << " " << constants::ELEMENT_SYMBOLS.at(atomic_num - 1) << ", "
                  << std::fixed << std::setprecision(4) << value << "\n";
    }
    std::cout << std::defaultfloat << std::flush;
}

void exitWithConfigOptionError(
    const std::string& name,
    const std::string& type_name,
    const std::exception& err
) {
    std::cerr << "ERROR (libtorchani)\n"
              << "Could not get config option " << name << " as " << type_name
              << ": " << err.what() << std::endl;
    std::exit(2);
}

void disableTorchReducedFloatPrecisionOptimizations() {
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
    at::globalContext().setAllowFP16ReductionCuBLAS(false);
    LOG(INFO) << "Disabled TF32 in cuBLAS and cuDNN, and FP16 reduction in cuBLAS";
}

// Return the directory of *this *.so*
std::string libtorchaniDir() {
    // &libtorchaniDir is the memory address of *this function*
    // dladdr searches for the *.so that contains this memory address
    // the Dl_info struct containes the absolute path to the *.so
    // dladdr depends on a cast of a function pointer to void*. This is
    // fully sound only in POSIX
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&libtorchaniDir), &info) == 0) {
        // dladdr returns 0 if it can't find the *.so that contains the requested addr
        throw std::runtime_error("Could not locate the libtorchani.so install directory");
    }
    std::string lib_path = info.dli_fname;
    return lib_path.substr(0, lib_path.find_last_of("/"));
}

// One float per atom, from a newline separated file. Exits if the file can't be read,
// or if the number of floats doesn't match num_atoms
torch::Tensor readExtraFeatures(const std::string& fpath, long num_atoms) {
    std::ifstream file(fpath);
    if (!file) {
        std::cerr << "ERROR (libtorchani)\n"
                  << "Could not open the extra features file " << fpath << std::endl;
        std::exit(2);
    }
    std::vector<double> extra_features;
    double value = 0.0;
    while (file >> value) {
        extra_features.push_back(value);
    }
    if (!file.eof()) {
        std::cerr << "ERROR (libtorchani)\n"
                  << "The extra features file " << fpath << " has values that are not floats"
                  << std::endl;
        std::exit(2);
    }
    if (static_cast<long>(extra_features.size()) != num_atoms) {
        std::cerr << "ERROR (libtorchani)\n"
                  << "The extra features file " << fpath << " has "
                  << extra_features.size() << " values, but there are " << num_atoms
                  << " atoms" << std::endl;
        std::exit(2);
    }
    return torch::tensor(extra_features, torch::TensorOptions().dtype(torch::kDouble));
}

void setModelExtraFeatures(torch::Tensor extra_features) {
    try {
        model->setExtraFeatures(extra_features);
    } catch (const std::runtime_error& e) {
        std::cerr << "ERROR (libtorchani)\n"
                  << "Could not set the extra features of the model:\n"
                  << e.what() << std::endl;
        std::exit(2);
    }
}

void loadTorchaniExtension(const std::string& lib_name) {
    std::string lib_path = libtorchaniDir() + "/" + lib_name;
    // RTLD_NOW | RTLD_LOCAL are dlopen flags that control *when* the librarie's
    // symbols get resolved, and who else can see the symbols.
    // RTLD_LOCAL doesn't export the symbols globally
    // If dlopen fails to load with the passed flags, it returns nullptr
    if (dlopen(lib_path.c_str(), RTLD_NOW | RTLD_LOCAL) == nullptr) {
        throw std::runtime_error("Could not load " + lib_path + ": " + dlerror());
    }
    LOG(INFO) << "Loaded extension: " << lib_path;
}

// Store the outputs of the model other than the energy and the forces as
// "__result_<name>", without the molecules dim, replacing the results of previous
// calculations. The QBC and its grads are stored in kcal/mol
void storeResults(const mlip::MlipOutput& output) {
    tensor_container.clearStartingWith("__result_");
    for (const auto& [name, value] : output) {
        if (name == "energy" or name == "forces" or name == "mm_forces") {
            continue;
        }
        torch::Tensor result = value.detach();
        if (name == "qbc" or name == "qbc_grad") {
            result = result * constants::HARTREE_TO_KCALMOL;
        }
        if (result.dim() > 0 and result.size(0) == 1) {
            result = result.squeeze(0);
        }
        tensor_container.store("__result_" + name, result);
    }
}

// Output files of the write_* options. The forces are only the ones calculated by
// the MLIP Interface, they don't include contributions added later by the md engine
struct OutputFiles {
    std::string coords;
    std::string forces;
    std::string mm_forces;  // Empty if there is no MM region
    std::string atomic_charges;
    std::string atomic_volumes;
    std::string atomic_alphas;
    std::string coords_comment;
};

const OutputFiles OUTPUT_FILES{
    "coords.xyz",
    "forces.mlip.dat",
    "",
    "atomic_charges.mlip.dat",
    "atomic_volumes.mlip.dat",
    "atomic_alphas.mlip.dat",
    "MLIP: .xyz trajectory file",
};

const OutputFiles MLMM_OUTPUT_FILES{
    "ml_region.xyz",
    "forces_ml_region.mlip.dat",
    "forces_mlmm_mm_region.mlip.dat",
    "atomic_charges_ml_region.mlip.dat",
    "atomic_volumes_ml_region.mlip.dat",
    "atomic_alphas_ml_region.mlip.dat",
    "ML/MM: .xyz trajectory file of the ML region",
};

// Path of an output file, in the output_dir option, or the current working directory if
// it is empty
std::string outputPath(const std::string& fname) {
    const std::string& output_dir = global_config.getString("output_dir");
    if (output_dir.empty()) {
        return fname;
    }
    return (std::filesystem::path(output_dir) / fname).string();
}

// Create empty output files for the write_* options that are set, and output_dir if
// it doesn't exist
void createOutputFiles(const OutputFiles& files) {
    bool writes_outputs = global_config.getBool("write_coords")
        or global_config.getBool("write_forces")
        or global_config.getBool("write_atomic_charges")
        or global_config.getBool("write_atomic_volumes")
        or global_config.getBool("write_atomic_alphas");
    if (not writes_outputs) {
        return;
    }
    try {
        const std::string& output_dir = global_config.getString("output_dir");
        if (not output_dir.empty()) {
            std::filesystem::create_directories(output_dir);
        }
        if (global_config.getBool("write_coords")) {
            io::createEmptyFile(outputPath(files.coords));
        }
        if (global_config.getBool("write_forces")) {
            io::createEmptyFile(outputPath(files.forces));
            if (not files.mm_forces.empty()) {
                io::createEmptyFile(outputPath(files.mm_forces));
            }
        }
        if (global_config.getBool("write_atomic_charges")) {
            io::createEmptyFile(outputPath(files.atomic_charges));
        }
        if (global_config.getBool("write_atomic_volumes")) {
            io::createEmptyFile(outputPath(files.atomic_volumes));
        }
        if (global_config.getBool("write_atomic_alphas")) {
            io::createEmptyFile(outputPath(files.atomic_alphas));
        }
    } catch (const std::runtime_error& e) {
        std::cerr << "ERROR (libtorchani)\n" << e.what() << std::endl;
        std::exit(2);
    }
}

// Throws if the model doesn't predict the atomic scalars
void appendAtomicScalarsFrame(
    const mlip::MlipOutput& output, const std::string& key, const std::string& fname
) {
    if (output.count(key) == 0) {
        throw std::runtime_error(
            "write_" + key + " = T, but the model doesn't predict " + key
        );
    }
    io::appendDatFrame(outputPath(fname), output.at(key).view({-1}));
}

// Append the outputs of a calculation requested by the write_* options, if the current
// MD step is a multiple of write_interval, and they weren't written yet for it. Forces are in
// Hartree/Angstrom, and are written in kcal/mol/Angstrom. mm_forces is only written if
// files has an mm_forces file
void writeOutputs(
    const OutputFiles& files,
    const torch::Tensor& atomic_numbers,
    const torch::Tensor& coords,
    const torch::Tensor& forces,
    const torch::Tensor& mm_forces,
    const mlip::MlipOutput& output
) {
    bool is_write_step = current_md_step % global_config.getInt("write_interval") == 0
        and last_written_md_step != current_md_step;
    if (not is_write_step) {
        return;
    }
    last_written_md_step = current_md_step;
    try {
        if (global_config.getBool("write_coords")) {
            io::appendXyzFrame(
                outputPath(files.coords),
                atomic_numbers.view({-1}),
                coords.view({-1, 3}),
                files.coords_comment
            );
        }
        if (global_config.getBool("write_forces")) {
            io::appendDatFrame(
                outputPath(files.forces),
                forces.view({-1, 3}) * constants::HARTREE_TO_KCALMOL
            );
            if (not files.mm_forces.empty() and mm_forces.numel() > 0) {
                io::appendDatFrame(
                    outputPath(files.mm_forces),
                    mm_forces.view({-1, 3}) * constants::HARTREE_TO_KCALMOL
                );
            }
        }
        if (global_config.getBool("write_atomic_charges")) {
            appendAtomicScalarsFrame(output, "atomic_charges", files.atomic_charges);
        }
        if (global_config.getBool("write_atomic_volumes")) {
            appendAtomicScalarsFrame(output, "atomic_volumes", files.atomic_volumes);
        }
        if (global_config.getBool("write_atomic_alphas")) {
            appendAtomicScalarsFrame(output, "atomic_alphas", files.atomic_alphas);
        }
    } catch (const std::runtime_error& e) {
        std::cerr << "ERROR (libtorchani)\n" << e.what() << std::endl;
        std::exit(2);
    }
}

// Charges of the ML atoms stored by the md engine, used by the coupling if
// use_mlip_atomic_charges = F. Exits if they are missing, or don't match the ML atoms
torch::Tensor fixedAtomicCharges(int ml_atoms_num, const std::string& mlmm_coupling) {
    if (not tensor_container.contains("fixed_atomic_charges")) {
        std::cerr << "ERROR (libtorchani)\n"
                  << "mlmm_coupling = '" << mlmm_coupling << "' with"
                  << " use_mlip_atomic_charges = F uses fixed atomic charges for the ML"
                  << " atoms, they must be stored as 'fixed_atomic_charges' (with"
                  << " mlip_store_tensor) before running the calculation" << std::endl;
        std::exit(2);
    }
    torch::Tensor fixed_atomic_charges = tensor_container.get("fixed_atomic_charges");
    if (fixed_atomic_charges.numel() != ml_atoms_num) {
        std::cerr << "ERROR (libtorchani)\n"
                  << "The stored 'fixed_atomic_charges' have "
                  << fixed_atomic_charges.numel() << " values, but there are "
                  << ml_atoms_num << " ML atoms" << std::endl;
        std::exit(2);
    }
    return fixed_atomic_charges.reshape({ml_atoms_num});
}

void calcEnergyForceCustomMlmm(
    int ml_atoms_num,
    int mm_atoms_num,
    double ml_coords_buf[][3],
    double mm_coords_buf[][3],
    double mm_atomic_charges_buf[],
    double cell_buf[][3],
    bool use_pbc,
    bool verbose,
    /* outputs */
    double ml_forces_buf[][3],
    double mm_forces_buf[][3],
    double* pot_energy_buf
) {
    torch::Tensor atomic_numbers = tensor_container.get("__atomic_numbers");
    if (!model->supportsCustomMlmm()) {
        std::cerr << "ERROR (libtorchani)\n"
                  << "mlmm_coupling = 'custom' requires the model to export"
                  << " 'compute_mlmm(...)'. "
                  << " Consult the TorchANI-Amber README.md for the required signature\n"
                  << "https://github.com/roitberg-group/torchani-amber" << std::endl;
        std::exit(2);
    }
    torch::Tensor ml_coords = tensor_utils::dblBufToCoordsTensor(global_config, ml_coords_buf, ml_atoms_num);
    torch::Tensor mm_coords = tensor_utils::dblBufToCoordsTensor(global_config, mm_coords_buf, mm_atoms_num);
    torch::Tensor mm_charges = tensor_utils::dblBufToTensor(global_config, mm_atomic_charges_buf, {1, mm_atoms_num});
    torch::Tensor cell = tensor_utils::dblBufToCellTensor(global_config, cell_buf);

    std::optional<torch::Tensor> maybe_cell;
    if (use_pbc) {
        maybe_cell = cell;
    }
    torch::Tensor empty_pairs = torch::empty({2, 0}, ml_coords.options().dtype(torch::kLong));
    mlip::MlipOutput output = model->computeMlmm(
        atomic_numbers,
        ml_coords,
        mm_coords,
        mm_charges,
        /*pairs_12=*/empty_pairs,
        /*pairs_13=*/empty_pairs,
        /*pairs_14=*/empty_pairs,
        maybe_cell,
        global_config.getInt("ml_charge"),
        global_config.getInt("ml_spin_mult"),
        /*mm_cutoff=*/0.0,
        /*atomic=*/false,
        /*ensemble_values=*/false,
        /*molecule_idxs=*/std::nullopt,
        /*calc_forces=*/true
    );
    storeResults(output);
    torch::Tensor energy = output.at("energy");

    populateEmbeddingForces(
        output.at("forces"), output.at("mm_forces"), ml_atoms_num, mm_atoms_num,
        ml_forces_buf, mm_forces_buf
    );
    writeOutputs(
        MLMM_OUTPUT_FILES, atomic_numbers, ml_coords, output.at("forces"),
        output.at("mm_forces"), output
    );
    populateEnergyScalar(energy, pot_energy_buf);
    if (verbose) {
        std::cout << std::fixed << std::setprecision(4)
                  << "\n ML/MM: CUSTOM ML-MM COUPLING ENERGY (TOTAL) =" << std::setw(14)
                  << *pot_energy_buf << std::defaultfloat << std::endl;
    }
}

/**
 * Note that currently this function can't split the QM forces in two parts, it only
 * outputs net forces due to the model in vacuum and the potential energies
 *
 * Currently this is the only code path that supports charged molecules
 */
void calcEnergyForceMlmm(
    int ml_atoms_num,
    int mm_atoms_num,
    double ml_coords_buf[][3],
    double mm_coords_buf[][3],  // shape (mm_atoms_num, 3)
    double mm_atomic_charges_buf[],  // shape (mm_atoms_num,)
    double cell_buf[][3],  // shape (3, 3), only used with use_pbc
    bool use_pbc,  // Only supported with mlmm_coupling = 'custom'
    bool verbose,
    bool calc_qbc,
    /* outputs */
    double ml_forces_buf[][3],  // shape (ml_atoms_num, 3)
    double mm_forces_buf[][3],  // shape (mm_atoms_num, 3)
    double* pot_energy_buf
) {
    torch::Tensor atomic_numbers = tensor_container.get("__atomic_numbers");
    const std::string& mlmm_coupling = global_config.getString("mlmm_coupling");
    if (mlmm_coupling != "custom" && use_pbc) {
        std::cerr << "ERROR (libtorchani)\n"
                  << "use_pbc is only supported with mlmm_coupling = 'custom'" << std::endl;
        std::exit(2);
    }
    if (mlmm_coupling == "custom" && calc_qbc) {
        std::cerr << "ERROR (libtorchani)\n"
                  << "mlmm_coupling = 'custom' doesn't support the QBC" << std::endl;
        std::exit(2);
    }
    if (mlmm_coupling == "custom") {
        calcEnergyForceCustomMlmm(
            ml_atoms_num,
            mm_atoms_num,
            ml_coords_buf,
            mm_coords_buf,
            mm_atomic_charges_buf,
            cell_buf,
            use_pbc,
            verbose,
            ml_forces_buf,
            mm_forces_buf,
            pot_energy_buf
        );
        return;
    }
    torch::Tensor ml_coords = tensor_utils::dblBufToCoordsTensor(global_config, ml_coords_buf, ml_atoms_num);

    // Without coupling, the energy doesn't depend on the MM coords, so the model
    // calculates the forces on the ML atoms directly
    bool coupling_is_none = mlmm_coupling == "none";
    mlip::MlipOutput output = model->computeSinglePoint(
        atomic_numbers,
        ml_coords,
        /*cell=*/std::nullopt,
        global_config.getInt("ml_charge"),
        global_config.getInt("ml_spin_mult"),
        /*atomic=*/false,
        /*ensemble_values=*/false,
        /*molecule_idxs=*/std::nullopt,
        /*calc_forces=*/coupling_is_none,
        /*calc_qbc_and_qbc_grad=*/calc_qbc
    );
    storeResults(output);
    torch::Tensor ene_pot_invacuo = output.at("energy");

    // Embedding calculations are made in atomic units, so outputs are in Ha
    torch::Tensor ene_pot_coulomb =
        torch::zeros(1, torch::dtype(global_config.dtype()).device(global_config.device()));
    torch::Tensor ene_pot_pol =
        torch::zeros(1, torch::dtype(global_config.dtype()).device(global_config.device()));
    torch::Tensor ene_pot_dist =
        torch::zeros(1, torch::dtype(global_config.dtype()).device(global_config.device()));
    torch::Tensor ene_pot_total = ene_pot_invacuo;
    torch::Tensor ml_forces;
    torch::Tensor mm_forces;
    if (coupling_is_none) {
        ml_forces = output.at("forces");
        mm_forces = torch::zeros({1, mm_atoms_num, 3}, ml_coords.options());
    } else {
        torch::Tensor mm_coords =
            tensor_utils::dblBufToCoordsTensor(global_config, mm_coords_buf, mm_atoms_num);
        torch::Tensor mm_atomic_charges =
            tensor_utils::dblBufToTensor(global_config, mm_atomic_charges_buf, {mm_atoms_num});
        torch::Tensor ml_atomic_charges;
        if (global_config.getBool("use_mlip_atomic_charges")) {
            if (output.count("atomic_charges") == 0) {
                std::cerr << "ERROR (libtorchani)\n"
                          << "use_mlip_atomic_charges = T, but the model doesn't predict"
                          << " atomic charges" << std::endl;
                std::exit(2);
            }
            ml_atomic_charges = output.at("atomic_charges");
        } else {
            ml_atomic_charges = fixedAtomicCharges(ml_atoms_num, mlmm_coupling);
        }

        torch::Tensor delta = torch::reshape(mm_coords, {-1, 1, 3}) -
            torch::reshape(ml_coords, {1, -1, 3});
        torch::Tensor mm_to_ml_distances =
            torch::linalg_norm(delta, std::nullopt, 2);

        ene_pot_coulomb = electro::coulombicEmbeddingEnergy(
            ml_atomic_charges, mm_atomic_charges, mm_to_ml_distances
        );
        if (mlmm_coupling == "simple-polarizable") {
            // Exits if the ML/MM config was not read, so it can be read afterwards
            torch::Tensor atomic_alphas;
            if (global_config.getBool("use_mlip_atomic_volumes")) {
                if (output.count("atomic_volumes") == 0) {
                    std::cerr << "ERROR (libtorchani)\n"
                              << "use_mlip_atomic_volumes = T, but the model doesn't"
                              << " predict atomic volumes" << std::endl;
                    std::exit(2);
                }
                atomic_alphas = mlVolumeKzs() * output.at("atomic_volumes").view({-1});
            } else {
                atomic_alphas = mlAtomicPolarizabilities();
            }
            double inv_pol_dielectric = global_config.getDouble("inv_pol_dielectric");
            ene_pot_pol = electro::polarizableEmbeddingEnergy(
                ml_coords,
                atomic_alphas,
                mm_coords,
                mm_atomic_charges,
                mm_to_ml_distances,
                inv_pol_dielectric
            );
            ene_pot_dist = -0.5 * ene_pot_pol;
        }

        ene_pot_total = ene_pot_invacuo + ene_pot_dist + ene_pot_pol + ene_pot_coulomb;
        std::vector<torch::Tensor> energy_vec = {ene_pot_total.sum()};
        std::vector<torch::Tensor> coord_vec = {ml_coords, mm_coords};
        // the output is a vector of tensors, one per element of coord_vec, so the
        // [0] and [1] elements are the derivatives wrt the ML and MM coords, and their
        // negatives are the forces
        auto ad_output = torch::autograd::grad(
            energy_vec,
            coord_vec,
            /* grad_outputs= */ {},
            /* retain_graph= */ false,
            /* create_graph= */ false
        );
        ml_forces = -ad_output[0];
        mm_forces = -ad_output[1];
    }
    populateEmbeddingForces(
        ml_forces,
        mm_forces,
        ml_atoms_num,
        mm_atoms_num,
        ml_forces_buf,
        mm_forces_buf
    );
    writeOutputs(MLMM_OUTPUT_FILES, atomic_numbers, ml_coords, ml_forces, mm_forces, output);
    populateEnergyScalar(ene_pot_total, pot_energy_buf);
    if (verbose) {
        double ene_pot_invacuo_kcal = 0.0;
        double ene_pot_pol_kcal = 0.0;
        double ene_pot_dist_kcal = 0.0;
        double ene_pot_coulomb_kcal = 0.0;
        populateEnergyScalar(ene_pot_invacuo, &ene_pot_invacuo_kcal);
        populateEnergyScalar(ene_pot_pol, &ene_pot_pol_kcal);
        populateEnergyScalar(ene_pot_dist, &ene_pot_dist_kcal);
        populateEnergyScalar(ene_pot_coulomb, &ene_pot_coulomb_kcal);
        double ene_pot_embed_kcal = ene_pot_coulomb_kcal + ene_pot_dist_kcal + ene_pot_pol_kcal;
        std::cout << std::fixed << std::setprecision(4)
                  << " ML/MM: IN VACUO ENERGY = " << std::setw(14) << ene_pot_invacuo_kcal
                  << "\n ML/MM: ML-MM COUPLING ENERGY (COULOMBIC) =" << std::setw(14)
                  << ene_pot_coulomb_kcal
                  << "\n ML/MM: ML-MM COUPLING ENERGY (DISTORTION) =" << std::setw(14)
                  << ene_pot_dist_kcal
                  << "\n ML/MM: ML-MM COUPLING ENERGY (POLARIZATION) =" << std::setw(14)
                  << ene_pot_pol_kcal
                  << "\n ML/MM: ML-MM COUPLING ENERGY (TOTAL) =" << std::setw(14) << ene_pot_embed_kcal
                  << std::defaultfloat << std::endl;
    }
}

// Between mlip_calc_energy_atomic_charges_retain and mlip_calc_energy_atomic_charges_vjp
void exitIfChargesGraphIsRetained(
    std::source_location loc = std::source_location::current()
) {
    bool charges_graph_is_retained = tensor_container.contains("__retain_coords")
        or tensor_container.contains("__retain_energy")
        or tensor_container.contains("__retain_atomic_charges");
    if (charges_graph_is_retained) {
        std::cerr << "ERROR (libtorchani)\n"
                  << loc.function_name() << " can't be called between"
                  << " mlip_calc_energy_atomic_charges_retain and"
                  << " mlip_calc_energy_atomic_charges_vjp" << std::endl;
        std::exit(2);
    }
}

// Names starting with "__" are reserved for tensors stored internally
void exitIfNameIsReserved(const char* name) {
    if (std::string_view{name}.starts_with("__")) {
        std::cerr << "ERROR (libtorchani)\n"
                  << "Tensor " << name << " can't be stored, names starting with __ are"
                  << " reserved" << std::endl;
        std::exit(2);
    }
}

// Shape of a tensor stored through the C API. Exits if any dim, or dims_num, is negative
std::vector<long> shapeFromDims(const char* name, int* dims, int dims_num) {
    std::vector<long> shape;
    if (dims_num >= 0) {
        shape.assign(dims, dims + dims_num);
    }
    bool has_negative_dim = std::any_of(shape.begin(), shape.end(), [](long d) { return d < 0; });
    if (dims_num < 0 || has_negative_dim) {
        std::cerr << "ERROR (libtorchani)\n"
                  << "Tensor " << name << " can't be stored, dims_num and its dims must be"
                  << " non-negative" << std::endl;
        std::exit(2);
    }
    return shape;
}

}  // namespace

extern "C" {
void mlip_initialize(
    int num_atoms,
    int atomic_nums[],  // Shape (num_atoms,)
    const char* config_path,
    bool read_mlmm_config,
    const char* calling_program,
    bool print_config_info
) {
    // (0) Clear tensors stored in previous runs of the MLIP Interface
    tensor_container.clear();

    // (1) Read and print configuration, fail immediatly if running on an unsupported device
    try {
        global_config = config::GlobalConfig::fromNamelistFile(
            config_path, read_mlmm_config, calling_program
        );
    } catch (const config::unsupported_device_error& e) {
        std::cerr << "ERROR (libtorchani)\n" << e.what() << std::endl;
        std::exit(2);
    } catch (const std::exception& e) {
        std::cerr << "ERROR (libtorchani)\n"
                  << "Could not read the config from " << config_path << ":\n"
                  << e.what() << std::endl;
        std::exit(2);
    }
    if (print_config_info) {
        global_config.printInfo();
    }

    // (2) Disable TF32 and FP16 for accuracy
    disableTorchReducedFloatPrecisionOptimizations();

    // It is VERY important to get types correctly here. If the
    // types don't match then from_blob will not interpret the pointers correctly
    // and as a consequence the tensor will have junk memory inside. If the blob
    // is int then the tensor should be torch::kInt. If the blob is double
    // then the tensor should be torch::kDouble
    //
    // PBC tensor is initialized even if pbc is not needed afterwards,
    // since the tensor occupies almost no space
    torch::Tensor atomic_numbers = torch::from_blob(
        atomic_nums, {num_atoms}, torch::TensorOptions().dtype(torch::kInt)
    );

    atomic_numbers = atomic_numbers.to(global_config.device());
    atomic_numbers = atomic_numbers.to(torch::kLong);
    // This is necessary since torch has to use this tensor to index internally,
    // and only long tensors can be used for this, so widening of the ints has
    // to be performed.
    // Also, torchani needs an extra dimension as batch dimension
    atomic_numbers = atomic_numbers.unsqueeze(0);
    LOG(INFO) << "Initialized MLIP with atomic numbers:\n" << atomic_numbers;
    // All atoms are evaluated until mlip_set_no_eval_atoms_mask is called
    tensor_container.store("__all_atomic_numbers", atomic_numbers);
    tensor_container.store("__atomic_numbers", atomic_numbers);

    // (3) Load compiled extensions
    std::vector<std::string> extension_names = {"cuaev.so", "cell_list.so"};
    for (const std::string& lib_name : extension_names) {
        try {
            loadTorchaniExtension(lib_name);
        } catch (const std::runtime_error& e) {
            std::cerr << "WARNING (libtorchani)\n"
                      << e.what() << "\n"
                      << "Models that support this extension will fail to load,\n"
                      << "even if the extension is not called during execution"
                      << std::endl;
        }
    }

    // (4) Resolve model path
    std::string model_type = global_config.getString("model_type");
    std::filesystem::path model_path;
    if (std::find(
            constants::builtin_models.begin(),
            constants::builtin_models.end(),
            model_type
        ) != constants::builtin_models.end()) {
        std::string jit_model_fname = model_type + ".pt";
        //  AimNet2 models, and Nutmeg models, are stored in subdirs
        std::filesystem::path model_dir =
            std::filesystem::path{libtorchaniDir()} / TORCHANI_MLIPS_RELATIVE_DIR;
        if (model_type.starts_with("aimnet2")) {
            model_dir /= "aimnet2";
        } else if (model_type.starts_with("nutmeg")) {
            model_dir /= "nutmeg";
        }
        model_path = model_dir / jit_model_fname;
        LOG(INFO) << "MLIP Model file name: " << jit_model_fname;
    } else {
        // Assume the path has been passed directly
        model_path = model_type;
    }
    LOG(INFO) << "MLIP Model full path: " << model_path.string();

    // (5) Load model
    // The model is loaded from a pytorch file, which can be either a TorchScript
    // compiled file (*.pt) or an AOTInductor-compiled graph (*.pt2 file)
    try {
        model = mlip::MlipModel::from_model_path(
            model_path.string(), global_config.device(), global_config.dtype()
        );
    } catch (const std::exception& e) {
        std::cerr
            << "ERROR (libtorchani)\n"
            << "Could not load model correctly from path: " << model_path.string() << "\n"
            << "Make sure the path is absolute, and correctly specified,\n"
            << "or that you are correctly specifying one of the built-in models.\n"
            << "Also make sure the model has been correctly packaged\n"
            << std::endl;
        throw;
    }

    // (6) Set the correct model configuration
    int model_index = global_config.getInt("model_index");
    if (model_index != -1) {
        try {
            model->useOnlyEnsembleMember(model_index);
        } catch (const std::runtime_error& e) {
            std::cerr << "ERROR (libtorchani)\n"
                      << "You set 'model_index' to a value != -1, but:\n"
                      << e.what() << std::endl;
            std::exit(2);
        }
    }
    std::string strat = global_config.getBool("use_cuaev") ? "cuaev" : "pyaev";
    try {
        model->setStrategy(strat);
    } catch (const std::runtime_error& e) {
        std::cerr << "WARNING (libtorchani)\n"
                  << e.what() << "\n"
                  << "The passed strategy was " << strat
                  << ". The cuAEV extension will be disabled.\n"
                  << " If you are not using a traditional ANI model with AEVs,"
                  << " you can safely disregard this warning."
                  << std::endl;
    }
    // Models can read arbitrary features from a newline separated
    // file with floats (e.g. Nutmeg potentials use this)
    // Relative paths are resolved wrt the current working directory
    std::string extra_features_fpath = global_config.getString("extra_features_file_path");
    if (not extra_features_fpath.empty()) {
        extra_features_fpath = std::filesystem::absolute(extra_features_fpath).string();
        torch::Tensor extra_features = readExtraFeatures(extra_features_fpath, num_atoms);
        tensor_container.store("__all_extra_features", extra_features);
        tensor_container.store("__extra_features", extra_features);
        setModelExtraFeatures(extra_features);
    }

    createOutputFiles(read_mlmm_config ? MLMM_OUTPUT_FILES : OUTPUT_FILES);
    last_written_md_step.reset();

    // (7) Epilogue
    global_config.setMlipInterfaceIsInit(true);
    if (read_mlmm_config && global_config.getBool("use_mlip_atomic_volumes")) {
        // Exits if an element of the ML atoms has no kz
        mlVolumeKzs();
    }
    if (
        print_config_info
        && read_mlmm_config
        && global_config.getString("mlmm_coupling") == "simple-polarizable"
    ) {
        printPolarizabilitiesInUse();
    }
    LOG(INFO) << "Loaded MLIP model and finished MLIP Interface initialization";
}

void mlip_set_no_eval_atoms_mask(int num_atoms, int values[]) {
    exitIfNotInitialized(global_config);
    exitIfChargesGraphIsRetained();
    torch::Tensor all_atomic_numbers = tensor_container.get("__all_atomic_numbers");
    if (num_atoms != all_atomic_numbers.size(1)) {
        std::cerr << "ERROR (libtorchani)\n"
                  << "The no-eval atoms mask has " << num_atoms << " values, but there are "
                  << all_atomic_numbers.size(1) << " atoms" << std::endl;
        std::exit(2);
    }
    torch::Tensor no_eval_atoms_mask =
        torch::from_blob(values, {num_atoms}, torch::TensorOptions().dtype(torch::kInt));
    if (not ((no_eval_atoms_mask == 0) | (no_eval_atoms_mask == 1)).all().item<bool>()) {
        std::cerr << "ERROR (libtorchani)\n"
                  << "The no-eval atoms mask can only have values 0 and 1" << std::endl;
        std::exit(2);
    }
    no_eval_atoms_mask = no_eval_atoms_mask.to(torch::kBool).to(global_config.device());
    global_config.setNoEvalAtoms(no_eval_atoms_mask.any().item<bool>());
    torch::Tensor do_eval_atoms_idxs = (~no_eval_atoms_mask).nonzero().view({-1});
    torch::Tensor atomic_numbers = all_atomic_numbers.index_select(1, do_eval_atoms_idxs);
    LOG(INFO) << "After discarding unused atoms, atomic numbers are:\n" << atomic_numbers;
    tensor_container.store("__do_eval_atoms_idxs", do_eval_atoms_idxs);
    tensor_container.store("__atomic_numbers", atomic_numbers);
    if (tensor_container.contains("__all_extra_features")) {
        torch::Tensor extra_features = tensor_container.get("__all_extra_features")
                                           .index_select(0, do_eval_atoms_idxs.to(torch::kCPU));
        tensor_container.store("__extra_features", extra_features);
        setModelExtraFeatures(extra_features);
    }
}

void mlip_update_md_step(int md_step) {
    current_md_step = md_step;
}

void mlip_set_ml_charge_and_spin_mult(int charge, int spin_mult) {
    exitIfNotInitialized(global_config);
    exitIfChargesGraphIsRetained();
    global_config.setMlChargeAndSpinMult(charge, spin_mult);
}

void mlip_calc_energy_force_from_external_neighbors(
    int num_atoms,
    int num_neighbors,
    double coords_buf[][3],
    int neighborlist_buf[],
    double shifts_buf[][3],
    int* molecule_idxs_buf,  // Shape (num_atoms,)
    bool calc_only_bonded,
    /* outputs */
    double forces_buf[][3],
    double* pot_energy_buf
) {
    exitIfNotInitialized(global_config);
    torch::Tensor atomic_numbers = tensor_container.get("__atomic_numbers");
    timing::ScopedTimer timer{};
    if (!model->supportsComputeFromExternalNeighbors()) {
        std::cerr << "ERROR (libtorchani)\n"
                  << "To use an external neighborlist (use_amber_neighborlist=.true.)\n"
                  << " the model must export 'compute_from_external_neighbors(...)'."
                  << " Consult the TorchANI-Amber README.md for the required signature\n"
                  << "https://github.com/roitberg-group/torchani-amber" << std::endl;
        std::exit(2);
    }
    torch::Tensor coords = tensor_utils::dblBufToCoordsTensor(global_config, coords_buf, num_atoms);
    torch::Tensor shifts = tensor_utils::dblBufToShiftsTensor(global_config, shifts_buf, num_neighbors);
    torch::Tensor neighborlist =
        tensor_utils::intBufToNeighborlistTensor(global_config, neighborlist_buf, num_neighbors);
    torch::Tensor molecule_idxs =
        tensor_utils::intBufToI64Tensor(global_config, molecule_idxs_buf, {num_atoms});

    if (global_config.useNoEvalAtoms()) {
        torch::Tensor do_eval_atoms_idxs = tensor_container.get("__do_eval_atoms_idxs");
        torch::Tensor reduced_idxs = torch::full({num_atoms}, -1, neighborlist.options());
        reduced_idxs.index_put_(
            {do_eval_atoms_idxs},
            torch::arange(do_eval_atoms_idxs.size(0), neighborlist.options())
        );
        neighborlist = reduced_idxs.index({neighborlist});
        auto do_eval_pairs = (neighborlist >= 0).all(0).nonzero().flatten();
        neighborlist = neighborlist.index_select(1, {do_eval_pairs});
        shifts = shifts.index_select(0, {do_eval_pairs});
        molecule_idxs = molecule_idxs.index({do_eval_atoms_idxs});
        coords = coords.index_select(1, {do_eval_atoms_idxs});
    }

    std::optional<torch::Tensor> maybe_molecule_idxs;
    if (calc_only_bonded) {
        maybe_molecule_idxs = molecule_idxs;
    }
    mlip::MlipOutput output = model->computeFromExternalNeighbors(
        atomic_numbers,
        coords,
        neighborlist,
        shifts,
        global_config.getInt("ml_charge"),
        global_config.getInt("ml_spin_mult"),
        /*atomic=*/false,
        /*ensemble_values=*/false,
        maybe_molecule_idxs,
        /*calc_forces=*/true
    );
    storeResults(output);
    torch::Tensor energy = output.at("energy");
    populateForces(output.at("forces"), forces_buf, num_atoms);
    populateEnergyScalar(energy, pot_energy_buf);
    writeOutputs(
        OUTPUT_FILES, atomic_numbers, coords, output.at("forces"), torch::Tensor{}, output
    );
}

void mlip_calc_energy_force(
    int num_atoms,
    double coords_buf[][3],
    double cell_buf[][3],
    bool use_pbc,
    int* molecule_idxs_buf,
    bool calc_only_bonded,
    double forces_buf[][3],
    double* pot_energy_buf
) {
    exitIfNotInitialized(global_config);
    torch::Tensor atomic_numbers = tensor_container.get("__atomic_numbers");
    timing::ScopedTimer timer{};
    torch::Tensor coords = tensor_utils::dblBufToCoordsTensor(global_config, coords_buf, num_atoms);
    torch::Tensor cell = tensor_utils::dblBufToCellTensor(global_config, cell_buf);
    torch::Tensor molecule_idxs =
        tensor_utils::intBufToI64Tensor(global_config, molecule_idxs_buf, {num_atoms});
    // Atoms excluded from the evaluation feel no forces, and don't contribute to the
    // energy
    if (global_config.useNoEvalAtoms()) {
        torch::Tensor do_eval_atoms_idxs = tensor_container.get("__do_eval_atoms_idxs");
        molecule_idxs = molecule_idxs.index({do_eval_atoms_idxs});
        coords = coords.index_select(1, {do_eval_atoms_idxs});
    }
    // Inputs are setup with PBC
    std::optional<torch::Tensor> maybe_cell;
    if (use_pbc) {
        maybe_cell = cell;
    }
    std::optional<torch::Tensor> maybe_molecule_idxs;
    if (calc_only_bonded) {
        maybe_molecule_idxs = molecule_idxs;
    }
    mlip::MlipOutput output = model->computeSinglePoint(
        atomic_numbers,
        coords,
        maybe_cell,
        global_config.getInt("ml_charge"),
        global_config.getInt("ml_spin_mult"),
        /*atomic=*/false,
        /*ensemble_values=*/false,
        maybe_molecule_idxs,
        /*calc_forces=*/true
    );
    storeResults(output);
    torch::Tensor energy = output.at("energy");
    populateForces(output.at("forces"), forces_buf, num_atoms);
    populateEnergyScalar(energy, pot_energy_buf);
    writeOutputs(
        OUTPUT_FILES, atomic_numbers, coords, output.at("forces"), torch::Tensor{}, output
    );
}

void mlip_calc_energy_force_mlmm(
    int ml_atoms_num,
    int mm_atoms_num,
    double ml_coords_buf[][3],
    double mm_coords_buf[][3],
    double mm_atomic_charges_buf[],
    double cell_buf[][3],
    bool use_pbc,
    bool verbose,
    /* outputs */
    double ml_forces_buf[][3],
    double mm_forces_buf[][3],
    double* pot_energy_buf
) {
    exitIfNotInitialized(global_config);
    exitIfNoMlmmConfig(global_config);
    exitIfNoEvalAtoms(global_config);
    calcEnergyForceMlmm(
        ml_atoms_num,
        mm_atoms_num,
        ml_coords_buf,
        mm_coords_buf,
        mm_atomic_charges_buf,
        cell_buf,
        use_pbc,
        verbose,
        /*calc_qbc=*/false,
        ml_forces_buf,
        mm_forces_buf,
        pot_energy_buf
    );
}

void mlip_calc_energy_force_mlmm_qbc(
    int ml_atoms_num,
    int mm_atoms_num,
    double ml_coords_buf[][3],
    double mm_coords_buf[][3],
    double mm_atomic_charges_buf[],
    double cell_buf[][3],
    bool use_pbc,
    bool verbose,
    /* outputs */
    double ml_forces_buf[][3],
    double mm_forces_buf[][3],
    double* pot_energy_buf
) {
    exitIfNotInitialized(global_config);
    exitIfNoMlmmConfig(global_config);
    exitIfNoEvalAtoms(global_config);
    calcEnergyForceMlmm(
        ml_atoms_num,
        mm_atoms_num,
        ml_coords_buf,
        mm_coords_buf,
        mm_atomic_charges_buf,
        cell_buf,
        use_pbc,
        verbose,
        /*calc_qbc=*/true,
        ml_forces_buf,
        mm_forces_buf,
        pot_energy_buf
    );
}

void mlip_calc_energy_atomic_charges_retain(
    int num_atoms,
    double coords_buf[][3],
    /* outputs */
    double atomic_charges_buf[],
    double* pot_energy_buf
) {
    exitIfNotInitialized(global_config);
    exitIfNoEvalAtoms(global_config);
    torch::Tensor atomic_numbers = tensor_container.get("__atomic_numbers");
    // The graph outlives this call, so it must not refer to the caller's buffer
    // (dblBufToCoordsTensor does not copy for CPU double precision)
    torch::Tensor coords = tensor_utils::dblBufToCoordsTensor(global_config, coords_buf, num_atoms)
                               .detach()
                               .clone()
                               .requires_grad_(true);
    mlip::MlipOutput output = model->computeSinglePoint(
        atomic_numbers,
        coords,
        /*cell=*/std::nullopt,
        global_config.getInt("ml_charge"),
        global_config.getInt("ml_spin_mult"),
        /*atomic=*/false,
        /*ensemble_values=*/false,
        /*molecule_idxs=*/std::nullopt,
        /*calc_forces=*/false
    );
    if (output.count("atomic_charges") == 0) {
        std::cerr << "ERROR (libtorchani)\n"
                  << "mlip_calc_energy_atomic_charges_retain requires a model that predicts"
                  << " atomic charges" << std::endl;
        std::exit(2);
    }
    storeResults(output);
    torch::Tensor energy = output.at("energy");
    torch::Tensor atomic_charges = output.at("atomic_charges");

    populateEnergyScalar(energy, pot_energy_buf);
    // populateAtomicCharges converts its argument to CPU double, keep the
    // original tensor (and its graph) for the backward pass
    torch::Tensor charges_copy = atomic_charges;
    populateAtomicCharges(charges_copy, atomic_charges_buf, num_atoms);
    tensor_container.store("__retain_coords", coords);
    tensor_container.store("__retain_energy", energy);
    tensor_container.store("__retain_atomic_charges", atomic_charges);
}

void mlip_calc_energy_atomic_charges_vjp(
    int num_atoms,
    double ml_electrostatic_potential_buf[],
    /* outputs */
    double forces_buf[][3]
) {
    exitIfNotInitialized(global_config);
    bool charges_graph_is_retained = tensor_container.contains("__retain_coords")
        and tensor_container.contains("__retain_energy")
        and tensor_container.contains("__retain_atomic_charges");
    if (not charges_graph_is_retained) {
        std::cerr << "ERROR (libtorchani)\n"
                  << "mlip_calc_energy_atomic_charges_vjp must follow a call to"
                  << " mlip_calc_energy_atomic_charges_retain" << std::endl;
        std::exit(2);
    }
    torch::Tensor ml_electrostatic_potential = tensor_utils::dblBufToTensor(
        global_config, ml_electrostatic_potential_buf, {1, num_atoms}
    );
    std::vector<torch::Tensor> output_vec = {
        tensor_container.get("__retain_energy").sum() * constants::HARTREE_TO_KCALMOL +
        (tensor_container.get("__retain_atomic_charges") * ml_electrostatic_potential).sum()
    };
    std::vector<torch::Tensor> coord_vec = {tensor_container.get("__retain_coords")};
    torch::Tensor forces = -torch::autograd::grad(
                               output_vec,
                               coord_vec,
                               /* grad_outputs= */ {},
                               /* retain_graph= */ false
                           )[0];
    forces = forces.to(torch::kCPU, torch::kDouble);
    auto forces_a = forces.accessor<double, 3>();
    for (int atom = 0; atom != num_atoms; ++atom) {
        for (int c = 0; c != 3; ++c) {
            forces_buf[atom][c] = forces_a[0][atom][c];
        }
    }
    tensor_container.clear("__retain_coords");
    tensor_container.clear("__retain_energy");
    tensor_container.clear("__retain_atomic_charges");
}

void mlip_store_float_tensor(
    const char* name,
    double* values,
    int* dims,
    int dims_num
) {
    exitIfNotInitialized(global_config);
    exitIfNameIsReserved(name);
    std::vector<long> shape = shapeFromDims(name, dims, dims_num);
    // dblBufToTensor doesn't copy values that are already double on the CPU, and the
    // stored tensor must not refer to the caller's buffer
    torch::Tensor tensor = tensor_utils::dblBufToTensor(global_config, values, shape).clone();
    tensor_container.store(name, tensor);
}

void mlip_store_int_tensor(
    const char* name,
    int* values,
    int* dims,
    int dims_num
) {
    exitIfNotInitialized(global_config);
    exitIfNameIsReserved(name);
    std::vector<long> shape = shapeFromDims(name, dims, dims_num);
    // intBufToI64Tensor always copies, since it converts the values to int64
    tensor_container.store(name, tensor_utils::intBufToI64Tensor(global_config, values, shape));
}

void mlip_store_mask_tensor(
    const char* name,
    int* values,
    int* dims,
    int dims_num
) {
    exitIfNotInitialized(global_config);
    exitIfNameIsReserved(name);
    std::vector<long> shape = shapeFromDims(name, dims, dims_num);
    torch::Tensor tensor = torch::from_blob(values, shape, torch::TensorOptions().dtype(torch::kInt));
    if (not ((tensor == 0) | (tensor == 1)).all().item<bool>()) {
        std::cerr << "ERROR (libtorchani)\n"
                  << "Mask tensor " << name << " can only have values 0 and 1" << std::endl;
        std::exit(2);
    }
    // The cast to bool copies, so the stored tensor doesn't refer to the caller's buffer
    tensor_container.store(name, global_config.tensorToTorchaniDevice(tensor.to(torch::kBool)));
}

void mlip_fetch_result(const char* name, double* values, int* dims, int dims_num) {
    exitIfNotInitialized(global_config);
    std::string key = std::string("__result_") + name;
    if (not tensor_container.contains(key)) {
        std::cerr << "ERROR (libtorchani)\n"
                  << "There is no result '" << name << "', the last calculation didn't"
                  << " produce it" << std::endl;
        std::exit(2);
    }
    torch::Tensor result = tensor_container.get(key);
    std::vector<long> shape = shapeFromDims(name, dims, dims_num);
    if (result.sizes() != torch::IntArrayRef(shape)) {
        std::cerr << "ERROR (libtorchani)\n"
                  << "The result '" << name << "' has shape " << result.sizes()
                  << ", but the shape of the array is " << torch::IntArrayRef(shape)
                  << " (C order)" << std::endl;
        std::exit(2);
    }
    result = result.to(torch::kCPU, torch::kDouble).contiguous();
    std::copy(result.data_ptr<double>(), result.data_ptr<double>() + result.numel(), values);
}

void mlip_get_config_int_option(const char* name, int* out_value) {
    try {
        std::string key = lowercase(name);
        *out_value = global_config.getInt(key);
    } catch (const std::exception& err) {
        exitWithConfigOptionError(name, "an int", err);
    }
}

void mlip_get_config_bool_option(const char* name, bool* out_value) {
    try {
        std::string key = lowercase(name);
        *out_value = global_config.getBool(key);
    } catch (const std::exception& err) {
        exitWithConfigOptionError(name, "a bool", err);
    }
}

void mlip_get_config_float_option(const char* name, double* out_value) {
    try {
        std::string key = lowercase(name);
        *out_value = global_config.getDouble(key);
    } catch (const std::exception& err) {
        exitWithConfigOptionError(name, "a float", err);
    }
}

void mlip_get_config_str_option(
    const char* name,
    int out_value_len,
    /* outputs */
    char out_value[]
) {
    try {
        std::string key = lowercase(name);
        const std::string& value = global_config.getString(key);
        if (static_cast<int>(value.size()) + 1 > out_value_len) {
            throw std::length_error(
                "Value '" + value + "' doesn't fit in a buffer of length "
                + std::to_string(out_value_len)
            );
        }
        std::memcpy(out_value, value.c_str(), value.size() + 1);
    } catch (const std::exception& err) {
        exitWithConfigOptionError(name, "a str", err);
    }
}
}
