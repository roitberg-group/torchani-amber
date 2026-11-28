#pragma once

#include <string>

namespace utils {

// Print a warning to stderr, prefixed with "WARNING (mlip_interface)"
void printWarning(const std::string& msg);

// Print an error to stderr, prefixed with "ERROR (mlip_interface)"
void printError(const std::string& msg);

// Uses std::tolower, so letters are lowercased according to the C locale
std::string toLower(std::string str);

}  // namespace utils
