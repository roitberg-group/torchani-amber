#include <iostream>
#pragma once

#include <c10/core/ScalarType.h>
#include <torch/all.h>
#include <torch/types.h>


/**
 * Holds global TorchANI-Interface device and precision
 * */
class GlobalConfig {
    torch::TensorOptions m_opts;

    public:
    GlobalConfig(torch::DeviceType device = torch::kCPU, torch::DeviceIndex device_idx = -1, torch::ScalarType dtype = torch::kFloat):
        m_opts{torch::TensorOptions().device(torch::Device{device, device_idx}).dtype(dtype)}
    {}

    auto dtype() const -> torch::ScalarType {
        return m_opts.dtype().toScalarType();
    }

    auto device() const -> torch::DeviceType {
        return m_opts.device().type();
    }

    auto set_device_and_precision(bool use_cuda_device, torch::DeviceIndex device_idx, bool use_double_precision) {
        torch::DeviceType dev_type{use_cuda_device ? torch::kCUDA : torch::kCPU};
        torch::Device dev{torch::Device{dev_type, device_idx}};
        torch::ScalarType dtype{use_double_precision ? torch::kFloat64 : torch::kFloat32};
        m_opts = torch::TensorOptions().device(dev).dtype(dtype);
        return *this;
    }

    auto tensor_to_torchani_device(torch::Tensor x) const -> torch::Tensor {
        return x.to(m_opts.device());
    }

    auto tensor_to_torchani_dtype_and_device(torch::Tensor x) const -> torch::Tensor {
        return x.to(m_opts);
    }

    // Amber has PBC enabled in all directions for PME
    auto enabled_pbc() const -> torch::Tensor {
        return torch::tensor(
            {true, true, true},
            torch::TensorOptions()
                .dtype(torch::kBool)
                .device(m_opts.device())
        );
    }
};
