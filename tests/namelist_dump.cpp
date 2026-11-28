// Print the namelist groups of a file as JSON, used to test fortran_utils against f90nml
// Output: [{"name": "cntrl", "values": {"imin": [1], ...}}, ...]
#include <charconv>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <variant>

#include "fortran_utils.h"

std::string toJson(const std::string& str) {
    std::string json = "\"";
    for (char c : str) {
        if (c == '"' || c == '\\') {
            json += '\\';
            json += c;
        } else if (static_cast<unsigned char>(c) < 0x20) {
            char escaped[8];
            std::snprintf(escaped, sizeof(escaped), "\\u%04x", c);
            json += escaped;
        } else {
            json += c;
        }
    }
    return json + "\"";
}

// Reals always have a "." or exponent, so they are not read back as integers
std::string toJson(double real) {
    char buf[64];
    std::to_chars_result result = std::to_chars(buf, buf + sizeof(buf), real);
    std::string json(buf, result.ptr);
    if (json.find_first_of(".en") == std::string::npos) {
        json += ".0";
    }
    return json;
}

std::string toJson(const fortran_utils::NamelistValue& value) {
    if (const bool* boolean = std::get_if<bool>(&value)) {
        return *boolean ? "true" : "false";
    }
    if (const long* integer = std::get_if<long>(&value)) {
        return std::to_string(*integer);
    }
    if (const double* real = std::get_if<double>(&value)) {
        return toJson(*real);
    }
    if (const std::string* str = std::get_if<std::string>(&value)) {
        return toJson(*str);
    }
    return "null";
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <file>\n";
        return 1;
    }
    std::ifstream file(argv[1]);
    if (!file) {
        std::cerr << "Could not open " << argv[1] << "\n";
        return 1;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();

    std::string json = "[";
    for (
        const fortran_utils::NamelistGroup& group :
        fortran_utils::parseNamelists(buffer.str())
    ) {
        json += "{\"name\": " + toJson(group.name) + ", \"values\": {";
        for (const auto& [key, values] : group.values) {
            json += toJson(key) + ": [";
            for (const fortran_utils::NamelistValue& value : values) {
                json += toJson(value) + ", ";
            }
            json.erase(json.size() - 2);
            json += "], ";
        }
        if (!group.values.empty()) {
            json.erase(json.size() - 2);
        }
        json += "}}, ";
    }
    if (json.size() > 1) {
        json.erase(json.size() - 2);
    }
    std::cout << json << "]\n";
    return 0;
}
