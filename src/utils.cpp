#include "utils.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>

namespace utils {

void printWarning(const std::string& msg) {
    std::cerr << "WARNING (mlip_interface)\n" << msg << std::endl;
}

std::string toLower(std::string str) {
    std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    return str;
}

}  // namespace utils
