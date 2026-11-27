#include <torchani.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>

#include <unistd.h>

#include <torch/all.h>

#include "constants.h"
#include "mlip.h"

// c10 defines a CHECK macro, which must not replace the one from Catch2
#undef CHECK
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using mlip::MlipModel;
using mlip::ModelFormat;
using mlip::ModelOptions;

static std::string file_path = __FILE__;
static std::string python_models_dir =
    file_path.substr(0, file_path.find_last_of("/")) + "/python_models/";
static std::string ani1x_path = std::string(TORCHANI_MLIPS_BUILD_DIR) + "/ani1x.pt";

static auto tmpPath(const std::string& name) -> std::filesystem::path {
    return std::filesystem::temp_directory_path()
        / ("torchani_test_" + std::to_string(getpid()) + "_" + name);
}

// Initializing loads the compiled extensions that the TorchScript models need
static auto initializeWithAni1x() -> void {
    std::filesystem::path config_path = tmpPath("ani1x.in");
    std::ofstream(config_path) << "&mlip model_type = 'ani1x', use_cuda_device = F /\n";
    mlip_initialize(config_path.c_str(), "mlip", "none");
    std::filesystem::remove(config_path);
}

TEST_CASE("Model format string conversions", "[model_format]") {
    for (
        ModelFormat format :
        {ModelFormat::TORCHSCRIPT,
            ModelFormat::PT2_AOTI,
            ModelFormat::PYTHON,
            ModelFormat::AUTO}
    ) {
        CHECK(mlip::modelFormatFromString(mlip::modelFormatToString(format)) == format);
    }
    CHECK(mlip::modelFormatToString(ModelFormat::PT2_AOTI) == "pt2-aoti");
    CHECK_THROWS_AS(mlip::modelFormatFromString("pt2"), std::invalid_argument);
}

TEST_CASE("TorchScript model format", "[model_format]") {
    initializeWithAni1x();
    CHECK(
        MlipModel::from_model_path(ani1x_path, torch::kCPU, torch::kDouble)->format()
        == ModelFormat::TORCHSCRIPT
    );
    CHECK(
        MlipModel::from_model_path(
            ani1x_path, torch::kCPU, torch::kDouble, {}, ModelFormat::TORCHSCRIPT
        )
            ->format()
        == ModelFormat::TORCHSCRIPT
    );
    // AOTI packages are loaded lazily, so a mismatch fails on the first calculation
    std::unique_ptr<MlipModel> aoti_model = MlipModel::from_model_path(
        ani1x_path, torch::kCPU, torch::kDouble, {}, ModelFormat::PT2_AOTI
    );
    CHECK(aoti_model->format() == ModelFormat::PT2_AOTI);
    CHECK_THROWS(aoti_model->computeSinglePoint(
        torch::tensor({{1, 6}}, torch::kLong), torch::zeros({1, 2, 3}, torch::kDouble)
    ));
    CHECK_THROWS(
        MlipModel::from_model_path(
            ani1x_path, torch::kCPU, torch::kDouble, {}, ModelFormat::PYTHON
        )
    );
}

TEST_CASE("Python-only options are ignored for TorchScript models", "[model_format]") {
    initializeWithAni1x();
    ModelOptions options;
    options.skip_py_preamble = true;
    options.do_torch_compile = true;
    options.torch_compile_mode = "max-autotune";
    std::stringstream captured_cerr;
    std::streambuf* cerr_buf = std::cerr.rdbuf(captured_cerr.rdbuf());
    std::unique_ptr<MlipModel> model =
        MlipModel::from_model_path(ani1x_path, torch::kCPU, torch::kDouble, options);
    std::cerr.rdbuf(cerr_buf);
    CHECK(model->format() == ModelFormat::TORCHSCRIPT);
    CHECK_THAT(captured_cerr.str(), ContainsSubstring("skip_py_preamble only applies"));
    CHECK_THAT(
        captured_cerr.str(),
        ContainsSubstring("do_torch_compile and torch_compile_mode only apply")
    );
}

#ifndef TORCHANI_SUPPORT_PYTHON_MODELS
TEST_CASE("Python models are not supported", "[model_format]") {
    CHECK_THROWS_WITH(
        MlipModel::from_model_path(
            python_models_dir + "analytic", torch::kCPU, torch::kDouble
        ),
        ContainsSubstring("TORCHANI_SUPPORT_PYTHON_MODELS=OFF")
    );
}
#else

constexpr double K = 0.5;

static auto loadAnalyticModel(const ModelOptions& options = {})
    -> std::unique_ptr<MlipModel> {
    return MlipModel::from_model_path(
        python_models_dir + "analytic", torch::kCPU, torch::kDouble, options
    );
}

static auto testCoords() -> torch::Tensor {
    return torch::tensor({{{0.1, 0.2, 0.3}, {1.0, -1.0, 0.5}}}, torch::kDouble)
        .requires_grad_(true);
}

static auto squaredNorm(const torch::Tensor& coords) -> double {
    return (coords.detach() * coords.detach()).sum().item<double>();
}

static auto checkTensorsClose(const torch::Tensor& actual, const torch::Tensor& expect)
    -> void {
    CHECK(actual.sizes() == expect.sizes());
    CHECK(torch::allclose(actual, expect));
}

TEST_CASE("Python model single point", "[python_models]") {
    std::unique_ptr<MlipModel> model = loadAnalyticModel();
    CHECK(model->format() == ModelFormat::PYTHON);
    CHECK(model->supportsComputeFromExternalNeighbors());
    CHECK(model->supportsCustomMlmm());
    CHECK(model->supportsMultipoleMlmm());
    torch::Tensor species = torch::tensor({{1, 6}}, torch::kLong);
    torch::Tensor coords = testCoords();
    mlip::MlipOutput output = model->computeSinglePoint(
        species,
        coords,
        std::nullopt,
        /* net_charge= */ -1,
        /* spin_mult= */ 2,
        false,
        false,
        std::nullopt,
        /* calc_forces= */ true
    );
    CHECK_THAT(
        output.at("energy").item<double>(),
        WithinAbs(K * squaredNorm(coords) - 0.1 + 0.02, 1e-12)
    );
    checkTensorsClose(output.at("forces"), -2 * K * coords.detach());

    model->setExtraFeatures(torch::tensor({1.5, 0.5}, torch::kDouble));
    output = model->computeSinglePoint(species, coords);
    CHECK_THAT(
        output.at("energy").item<double>(),
        WithinAbs(K * squaredNorm(coords) + 0.01 + 2.0, 1e-12)
    );
    CHECK_NOTHROW(model->useOnlyEnsembleMember(0));
    CHECK_NOTHROW(model->setStrategy("cuaev"));
    CHECK_THROWS_WITH(
        model->setStrategy("unknown"),
        ContainsSubstring("doesn't accept the specified strategy")
    );
    CHECK_THROWS_WITH(
        model->computeSinglePoint(species, coords, std::nullopt, 0, 1, true),
        ContainsSubstring("doesn't support atomic=true")
    );
}

TEST_CASE("Python model from external neighbors", "[python_models]") {
    std::unique_ptr<MlipModel> model = loadAnalyticModel();
    torch::Tensor species = torch::tensor({{1, 6}}, torch::kLong);
    torch::Tensor coords = testCoords();
    torch::Tensor neighborlist = torch::tensor({{0, 0, 1}, {1, 1, 0}}, torch::kLong);
    torch::Tensor shifts = torch::zeros({3, 3}, torch::kDouble);
    mlip::MlipOutput output = model->computeFromExternalNeighbors(
        species,
        coords,
        neighborlist,
        shifts,
        0,
        1,
        false,
        false,
        std::nullopt,
        /* calc_forces= */ true
    );
    CHECK_THAT(
        output.at("energy").item<double>(),
        WithinAbs(K * squaredNorm(coords) + 0.003, 1e-12)
    );
    checkTensorsClose(output.at("forces"), -2 * K * coords.detach());
}

TEST_CASE("Python model ML/MM", "[python_models]") {
    std::unique_ptr<MlipModel> model = loadAnalyticModel();
    torch::Tensor species = torch::tensor({{1, 6}}, torch::kLong);
    torch::Tensor coords = testCoords();
    torch::Tensor mm_coords =
        torch::tensor({{{2.0, 0.0, 0.0}, {0.0, 3.0, 1.0}}}, torch::kDouble)
            .requires_grad_(true);
    torch::Tensor mm_charges = torch::tensor({{0.5, -0.25}}, torch::kDouble);
    double mm_energy = 0.5 * 4.0 - 0.25 * 10.0;
    torch::Tensor expect_mm_forces = -2 * mm_charges.unsqueeze(-1) * mm_coords.detach();

    mlip::MlipOutput output = model->computeMlmm(
        species,
        coords,
        mm_coords,
        mm_charges,
        std::nullopt,
        0,
        /* spin_mult= */ 2,
        /* mm_cutoff= */ 5.0,
        false,
        false,
        std::nullopt,
        /* calc_forces= */ true
    );
    CHECK_THAT(
        output.at("energy").item<double>(),
        WithinAbs(K * squaredNorm(coords) + mm_energy, 1e-12)
    );
    checkTensorsClose(output.at("forces"), -2 * K * coords.detach());
    checkTensorsClose(output.at("mm_forces"), expect_mm_forces);

    torch::Tensor mm_dipoles = torch::full({1, 2, 3}, 0.5, torch::kDouble);
    torch::Tensor mm_quads = torch::zeros({1, 2, 3, 3}, torch::kDouble);
    output = model->computeMultipoleMlmm(
        species,
        coords,
        mm_coords,
        mm_charges,
        mm_dipoles,
        mm_quads,
        std::nullopt,
        0,
        1,
        0.0,
        false,
        false,
        std::nullopt,
        /* calc_forces= */ true
    );
    CHECK_THAT(
        output.at("energy").item<double>(),
        WithinAbs(K * squaredNorm(coords) + mm_energy + 0.2 * 3.0, 1e-12)
    );
    checkTensorsClose(output.at("mm_forces"), expect_mm_forces);
    checkTensorsClose(output.at("atomic_charges"), species.to(torch::kDouble));
}

TEST_CASE("Python model preamble and spin_mult", "[python_models]") {
    std::string model_dir = python_models_dir + "no-spin-mult";
    torch::Tensor species = torch::tensor({{1, 6}}, torch::kLong);
    torch::Tensor coords = testCoords();

    // make_model returns a float32 model in training mode, fixed by the preamble
    std::unique_ptr<MlipModel> model =
        MlipModel::from_model_path(model_dir, torch::kCPU, torch::kDouble);
    mlip::MlipOutput output = model->computeSinglePoint(species, coords);
    CHECK(output.at("energy").dtype() == torch::kDouble);
    CHECK_THAT(
        output.at("energy").item<double>(), WithinAbs(squaredNorm(coords), 1e-12)
    );
    CHECK_THROWS_WITH(
        model->computeSinglePoint(species, coords, std::nullopt, 0, 3),
        ContainsSubstring("only spin_mult = 1 is supported")
    );

    ModelOptions options;
    options.skip_py_preamble = true;
    model = MlipModel::from_model_path(model_dir, torch::kCPU, torch::kDouble, options);
    output = model->computeSinglePoint(species, coords);
    CHECK_THAT(
        output.at("energy").item<double>(),
        WithinAbs(squaredNorm(coords) + 1000.0, 1e-4)
    );
}

TEST_CASE("Python model with legacy int charge", "[python_models]") {
    std::unique_ptr<MlipModel> model = MlipModel::from_model_path(
        python_models_dir + "legacy-charge", torch::kCPU, torch::kDouble
    );
    torch::Tensor species = torch::tensor({{1, 6}}, torch::kLong);
    torch::Tensor coords = testCoords();
    mlip::MlipOutput output =
        model->computeSinglePoint(species, coords, std::nullopt, /* net_charge= */ -2);
    CHECK_THAT(
        output.at("energy").item<double>(), WithinAbs(squaredNorm(coords) - 2.0, 1e-12)
    );
}

TEST_CASE("Python model with torch.compile", "[python_models]") {
    ModelOptions options;
    options.do_torch_compile = true;
    std::unique_ptr<MlipModel> model = loadAnalyticModel(options);
    torch::Tensor species = torch::tensor({{1, 6}}, torch::kLong);
    torch::Tensor coords = testCoords();
    mlip::MlipOutput output = model->computeSinglePoint(
        species, coords, std::nullopt, 0, 1, false, false, std::nullopt, true
    );
    CHECK_THAT(
        output.at("energy").item<double>(),
        WithinAbs(K * squaredNorm(coords) + 0.01, 1e-12)
    );
    checkTensorsClose(output.at("forces"), -2 * K * coords.detach());
}

TEST_CASE("Python model on CUDA", "[python_models]") {
    if (std::getenv("TORCHANI_SKIP_CUDA_TESTS")) {
        SKIP("CUDA tests disabled by TORCHANI_SKIP_CUDA_TESTS");
    }
    std::unique_ptr<MlipModel> model = MlipModel::from_model_path(
        python_models_dir + "analytic", torch::kCUDA, torch::kDouble
    );
    torch::Tensor species = torch::tensor({{1, 6}}, torch::kLong).to(torch::kCUDA);
    torch::Tensor coords = testCoords().detach().to(torch::kCUDA).requires_grad_(true);
    mlip::MlipOutput output = model->computeSinglePoint(
        species, coords, std::nullopt, 0, 1, false, false, std::nullopt, true
    );
    CHECK(output.at("forces").device().is_cuda());
    CHECK_THAT(
        output.at("energy").item<double>(),
        WithinAbs(K * squaredNorm(coords) + 0.01, 1e-12)
    );
}

// Every method returns sum(coords**2) (plus sum(mm_charges * mm_coords**2) for ML/MM)
// as the energies, and sum(coords, -1) as the atomic charges, in the given output form
static const char* output_forms_model = R"(import torch

FORM = "{form}"


def output(species, coords, extra_energies=0.0):
    energies = (coords**2).sum((-1, -2)) + extra_energies
    charges = coords.sum(-1)
    results = {"atomic_charges": charges}
    if FORM == "energies":
        return energies
    if FORM == "species-energies":
        return species, energies
    if FORM == "species-energies-charges":
        return species, energies, charges
    if FORM == "species-energies-results":
        return species, energies, results
    if FORM == "energies-results":
        return energies, results
    if FORM == "energies-charges":
        return energies, charges
    if FORM == "energies-none":
        return energies, None
    if FORM == "dict":
        return {"energies": energies, **results}
    raise ValueError(FORM)


class Model(torch.nn.Module):
    def forward(self, species_coordinates, charge=None, **kwargs):
        return output(*species_coordinates)

    def compute_from_external_neighbors(
        self, species, coords, neighbor_idxs, shifts, charge=None, **kwargs
    ):
        return output(species, coords)

    def compute_mlmm(self, species_coordinates, mm_coords, mm_charges, **kwargs):
        mm_energies = (mm_charges * (mm_coords**2).sum(-1)).sum(-1)
        return output(*species_coordinates, mm_energies)


def make_model(device, dtype):
    return Model()
)";

TEST_CASE("Python model output forms", "[python_models]") {
    std::string form = GENERATE(
        "energies",
        "species-energies",
        "species-energies-charges",
        "species-energies-results",
        "energies-results",
        "energies-charges",
        "energies-none",
        "dict"
    );
    INFO("Output form: " << form);
    bool form_has_charges =
        form != "energies" and form != "species-energies" and form != "energies-none";
    std::filesystem::path model_dir = tmpPath("output_forms_" + form);
    std::filesystem::create_directories(model_dir);
    std::string text = output_forms_model;
    text.replace(text.find("{form}"), 6, form);
    std::ofstream(model_dir / "model.py") << text;
    std::unique_ptr<MlipModel> model =
        MlipModel::from_model_path(model_dir.string(), torch::kCPU, torch::kDouble);

    torch::Tensor species = torch::tensor({{1, 6}}, torch::kLong);
    torch::Tensor coords = testCoords();
    torch::Tensor expect_charges = coords.detach().sum(-1);
    torch::Tensor neighborlist = torch::tensor({{0}, {1}}, torch::kLong);
    torch::Tensor mm_coords =
        torch::tensor({{{2.0, 0.0, 0.0}}}, torch::kDouble).requires_grad_(true);
    torch::Tensor mm_charges = torch::tensor({{0.5}}, torch::kDouble);

    mlip::MlipOutput forward_output = model->computeSinglePoint(
        species, coords, std::nullopt, 0, 1, false, false, std::nullopt, true
    );
    mlip::MlipOutput external_output = model->computeFromExternalNeighbors(
        species,
        coords,
        neighborlist,
        torch::zeros({1, 3}, torch::kDouble),
        0,
        1,
        false,
        false,
        std::nullopt,
        true
    );
    mlip::MlipOutput mlmm_output = model->computeMlmm(
        species,
        coords,
        mm_coords,
        mm_charges,
        std::nullopt,
        0,
        1,
        0.0,
        false,
        false,
        std::nullopt,
        true
    );
    for (const mlip::MlipOutput& output : {forward_output, external_output}) {
        CHECK_THAT(
            output.at("energy").item<double>(), WithinAbs(squaredNorm(coords), 1e-12)
        );
        checkTensorsClose(output.at("forces"), -2 * coords.detach());
    }
    CHECK_THAT(
        mlmm_output.at("energy").item<double>(),
        WithinAbs(squaredNorm(coords) + 0.5 * 4.0, 1e-12)
    );
    checkTensorsClose(mlmm_output.at("mm_forces"), -2 * 0.5 * mm_coords.detach());
    for (
        const mlip::MlipOutput& output : {forward_output, external_output, mlmm_output}
    ) {
        CHECK(output.count("atomic_charges") == (form_has_charges ? 1 : 0));
        if (form_has_charges) {
            checkTensorsClose(output.at("atomic_charges"), expect_charges);
        }
    }
    std::filesystem::remove_all(model_dir);
}

TEST_CASE("Python model loading errors", "[python_models]") {
    std::filesystem::path model_dir = tmpPath("python_model");
    std::filesystem::create_directories(model_dir);
    CHECK_THROWS_WITH(
        MlipModel::from_model_path(model_dir.string(), torch::kCPU, torch::kDouble),
        ContainsSubstring("Python model file not found")
    );

    std::ofstream(model_dir / "model.py")
        << "def make_model(device, dtype):\n"
        << "    raise ValueError('make_model failed on purpose')\n";
    CHECK_THROWS_WITH(
        MlipModel::from_model_path(model_dir.string(), torch::kCPU, torch::kDouble),
        ContainsSubstring("make_model failed on purpose")
    );

    std::ofstream(model_dir / "typing.py") << "\n";
    CHECK_THROWS_WITH(
        MlipModel::from_model_path(model_dir.string(), torch::kCPU, torch::kDouble),
        ContainsSubstring("Module 'typing'")
    );
    std::filesystem::remove_all(model_dir);
}

TEST_CASE("Python model through the C API", "[python_models]") {
    std::filesystem::path config_path = tmpPath("python_model.in");
    std::ofstream(config_path) << "&mlip model_type = '" << python_models_dir
                               << "analytic', model_format = 'python',\n"
                               << "  use_cuda_device = F, use_cuaev = F /\n";
    mlip_initialize(config_path.c_str(), "mlip", "none");
    std::filesystem::remove(config_path);
    int atomic_numbers[] = {1, 6};
    mlip_set_atomic_numbers(2, atomic_numbers, false);
    double coords[][3] = {{0.1, 0.2, 0.3}, {1.0, -1.0, 0.5}};
    double cell[][3] = {{10., 0., 0.}, {0., 10., 0.}, {0., 0., 10.}};
    int molecule_idxs[] = {1, 1};
    double forces[2][3];
    double pot_energy = 0.0;
    mlip_calc_energy_force(
        2, coords, cell, false, molecule_idxs, false, forces, &pot_energy
    );
    double squared_norm = 0.01 + 0.04 + 0.09 + 1.0 + 1.0 + 0.25;
    CHECK_THAT(
        pot_energy,
        WithinAbs((K * squared_norm + 0.01) * constants::HARTREE_TO_KCALMOL, 1e-9)
    );
    CHECK_THAT(
        forces[1][0], WithinAbs(-2 * K * 1.0 * constants::HARTREE_TO_KCALMOL, 1e-9)
    );
}
#endif
