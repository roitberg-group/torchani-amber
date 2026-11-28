#pragma once

#include <string>

namespace utils {

// Print a warning to stderr, prefixed with "WARNING (libtorchani)"
void printWarning(const std::string& msg);

// Uses std::tolower, so letters are lowercased according to the C locale
std::string toLower(std::string str);

}  // namespace utils
