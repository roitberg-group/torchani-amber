def Settings( **kwargs ):
  return {
    'flags': [ '-x', 'c++', '-Wall', '-Wextra', '-Werror' , '-std=c++14', '-I', 
        './lib/libtorch_cxx11abi_cu102_latest/include/torch/', 
        '-I', './include', 
        '-I', './lib/libtorch_cxx11abi_cu102_latest/include/torch/csrc/api/include/', 
        '-I', './lib/libtorch_cxx11abi_cu102_latest/include/torch/csrc/autograd/', 
        '-I', './lib/libtorch_cxx11abi_cu102_latest/include/', 
        ]
  }
#-I ./lib/libtorch_cxx11abi_cu110_1.7.0/include/
##-I ./lib/libtorch_cxx11abi_cu110_1.7.0/include/torch/
##-I ./lib/libtorch_cxx11abi_cu110_1.7.0/include/torch/csrc/autograd/
#-I ./lib/libtorch_cxx11abi_cu110_1.7.0/include/torch/csrc/api/include/
