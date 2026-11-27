#include "utils.h"

#include <iostream>
#include <string>

namespace utils {

void printWarning(const std::string& msg) {
    std::cerr << "WARNING (libtorchani)\n" << msg << std::endl;
}

}  // namespace utils
