#include "model_output.h"

#include <optional>
#include <utility>
#include <vector>

#include <torch/all.h>
#include <torch/csrc/autograd/autograd.h>

#include "mlip.h"

namespace model_output {

// Energies, extra outputs of the model, and optionally forces, QBC and QBC grads wrt
// the coords. If mm_coords is passed, the forces on the MM atoms are also calculated
auto computeOutput(
    torch::Tensor energy,
    mlip::MlipOutput extra_outputs,
    const torch::Tensor& coords,
    const std::optional<torch::Tensor>& mm_coords,
    bool calc_forces,
    bool calc_qbc_and_qbc_grad
) -> mlip::MlipOutput {
    mlip::MlipOutput output = std::move(extra_outputs);
    if (calc_qbc_and_qbc_grad) {
        torch::Tensor num_atoms = torch::tensor(
            coords.size(1),
            torch::TensorOptions().dtype(coords.dtype()).device(coords.device())
        );
        output["qbc"] = torch::std(energy, 0) / num_atoms.sqrt();
        energy = energy.mean(0);
    }
    output["energy"] = energy;

    // Old TorchANI outputs charges as (1, molecs, atoms), new as (molecs, atoms)
    if (output.count("atomic_charges") > 0 && output.at("atomic_charges").dim() == 3) {
        output["atomic_charges"] = output.at("atomic_charges").squeeze(0);
    }

    if (calc_forces) {
        bool retain_graph = calc_qbc_and_qbc_grad;
        if (mm_coords.has_value()) {
            std::vector<torch::Tensor> grads = torch::autograd::grad(
                {energy.sum()},
                {coords, *mm_coords},
                /* grad_outputs= */ {},
                retain_graph
            );
            output["forces"] = -grads[0];
            output["mm_forces"] = -grads[1];
        } else {
            output["forces"] = -torch::autograd::grad(
                {energy.sum()}, {coords}, /* grad_outputs= */ {}, retain_graph
            )[0];
        }
    }
    if (calc_qbc_and_qbc_grad) {
        output["qbc_grad"] = torch::autograd::grad(
            {output.at("qbc").sum()},
            {coords},
            /* grad_outputs= */ {},
            // Without calc_forces, the caller may differentiate the energy afterwards
            /* retain_graph= */ not calc_forces
        )[0];
    }
    return output;
}

}  // namespace model_output
