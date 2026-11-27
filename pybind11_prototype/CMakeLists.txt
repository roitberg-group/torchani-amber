cmake_minimum_required(VERSION 3.19)
project(PythonModelPrototype LANGUAGES CXX)

# Development.Embed provides Python.h and libpython3.X.so
find_package(Python REQUIRED COMPONENTS Interpreter Development.Embed)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import os, torch; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE TORCH_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

add_executable(run_python_model main.cpp python_model.cpp)
target_compile_features(run_python_model PRIVATE cxx_std_17)
# torch/include also holds the (header-only) pybind11 that torch was built with
target_include_directories(
    run_python_model PRIVATE "${TORCH_DIR}/include" "${TORCH_DIR}/include/torch/csrc/api/include"
)
target_link_directories(run_python_model PRIVATE "${TORCH_DIR}/lib")
target_link_libraries(run_python_model PRIVATE torch torch_cpu c10 torch_python Python::Python)
set_target_properties(run_python_model PROPERTIES BUILD_RPATH "${TORCH_DIR}/lib")
