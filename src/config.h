#pragma once

#include <c10/core/ScalarType.h>
#include <torch/all.h>
#include <torch/types.h>


/**
 * Holds global TorchANI-Interface device and precision
 * */
class GlobalConfig {
    torch::TensorOptions m_opts;
    bool m_use_no_eval_atoms;

    public:
    GlobalConfig(torch::DeviceType device = torch::kCPU, torch::DeviceIndex device_idx = -1, torch::ScalarType dtype = torch::kFloat, bool use_no_eval_atoms = false):
        m_opts{torch::TensorOptions().device(torch::Device{device, device_idx}).dtype(dtype)},
        m_use_no_eval_atoms{use_no_eval_atoms}
    {}

    auto dtype() const -> torch::ScalarType {
        return m_opts.dtype().toScalarType();
    }

    auto device() const -> torch::Device {
        return m_opts.device();
    }

    auto useNoEvalAtoms() const -> bool {
        return m_use_no_eval_atoms;
    }

    auto setNoEvalAtoms(bool use_no_eval_atoms) -> void {
        m_use_no_eval_atoms = use_no_eval_atoms;
    }

    auto setDeviceAndPrecision(bool use_cuda_device, torch::DeviceIndex device_idx, bool use_double_precision) -> void {
        torch::DeviceType dev_type{use_cuda_device ? torch::kCUDA : torch::kCPU};
        torch::Device dev{torch::Device{dev_type, device_idx}};
        torch::ScalarType dtype{use_double_precision ? torch::kFloat64 : torch::kFloat32};
        m_opts = torch::TensorOptions().device(dev).dtype(dtype);
    }

    auto tensorToTorchaniDevice(torch::Tensor x) const -> torch::Tensor {
        return x.to(m_opts.device());
    }

    auto tensorToTorchaniDtypeAndDevice(torch::Tensor x) const -> torch::Tensor {
        return x.to(m_opts);
    }

    // Amber has PBC enabled in all directions for PME
    auto enabledPbc() const -> torch::Tensor {
        return torch::tensor(
            {true, true, true},
            torch::TensorOptions()
                .dtype(torch::kBool)
                .device(m_opts.device())
        );
    }
};
