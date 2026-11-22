#pragma once

#include <map>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace fortran_utils {

// std::monostate represents a Fortran 'null' value (e.g. the second value in "a = 1,,3")
using NamelistValue = std::variant<std::monostate, bool, long, double, std::string>;

// Base class of all namelist parsing errors
class namelist_parse_error : public std::runtime_error {
    public:
    using std::runtime_error::runtime_error;
};

// Malformed input, e.g. a missing "=" after a key, or a bad group terminator
class namelist_syntax_error : public namelist_parse_error {
    public:
    using namelist_parse_error::namelist_parse_error;
};

// The text ends inside a group, before "/" or "&end"
class namelist_unterminated_group_error : public namelist_parse_error {
    public:
    using namelist_parse_error::namelist_parse_error;
};

class namelist_unterminated_string_error : public namelist_parse_error {
    public:
    using namelist_parse_error::namelist_parse_error;
};

// A value that is not a number or a logical, and is not quoted
class namelist_unquoted_string_error : public namelist_parse_error {
    public:
    using namelist_parse_error::namelist_parse_error;
};

// Valid Fortran not supported by the parser: indexed keys, derived types, complex values
class namelist_unsupported_feature_error : public namelist_parse_error {
    public:
    using namelist_parse_error::namelist_parse_error;
};

// A number, or a repeat count, that doesn't fit in its C++ type
class namelist_out_of_range_error : public namelist_parse_error {
    public:
    using namelist_parse_error::namelist_parse_error;
};

struct NamelistGroup {
    std::string name;
    std::map<std::string, std::vector<NamelistValue>> values;
};

/**
 * Parse all namelist groups in the text of a Fortran input file, in order of appearance
 *
 * A group starts with "&name" or "$name" as the first token of a line, and ends with
 * "/", "&end" or "$end". Text outside groups is ignored. Group names and keys are
 * lowercased. Types are inferred from the values, since they are not declared:
 * integers -> long, reals -> double, .true./.t./true/t (and false variants) -> bool,
 * quoted strings -> std::string. Scalars are stored as single-element vectors.
 *
 * Repeated keys are assigned element-wise, and null values don't overwrite existing
 * ones, as in Fortran. Unquoted strings are rejected, as gfortran does. Indexed keys
 * "a(1)", derived types "a%b", and complex values are not supported. All errors
 * throw a subclass of namelist_parse_error.
 *
 * If group_name is not empty, only the groups with that name (case-insensitive) are
 * parsed and returned. Other groups are skipped without parsing, as Fortran does, so
 * they can't cause errors.
 * */
std::vector<NamelistGroup> parseNamelists(
    const std::string& text,
    const std::string& group_name = ""
);

}  // namespace fortran_utils
