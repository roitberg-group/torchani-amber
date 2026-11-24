#pragma once

#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace fortran_utils {

// std::monostate represents a Fortran 'null' value (e.g. the second value in "a = 1,,3")
using NamelistValue = std::variant<std::monostate, bool, long, double, std::string>;

// FLOAT -> double, INT -> long, STR -> std::string, BOOL -> bool, INT_OR_STR -> long
// or std::string. Scalars take one value, VEC_* any number of values, and VEC3_*
// exactly 3
enum class NamelistType {
    FLOAT,
    INT,
    STR,
    BOOL,
    INT_OR_STR,
    VEC_FLOAT,
    VEC_INT,
    VEC_STR,
    VEC_BOOL,
    VEC3_FLOAT,
    VEC3_INT,
    VEC3_STR,
    VEC3_BOOL,
};

// Maps each key to the type of its values. Keys are case-insensitive
using NamelistSchema = std::unordered_map<std::string, NamelistType>;

// Maps each alias to the key it is normalized to. Aliases are case-insensitive
using NamelistAliases = std::unordered_map<std::string, std::string>;

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

// A value, or number of values, that doesn't match the type of its key in the schema
class namelist_type_error : public namelist_parse_error {
    public:
    using namelist_parse_error::namelist_parse_error;
};

// A key that is not in the schema
class namelist_unknown_key_error : public namelist_parse_error {
    public:
    using namelist_parse_error::namelist_parse_error;
};

// Two different names of the same key in one group, e.g. a key and its alias
class namelist_alias_conflict_error : public namelist_parse_error {
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
 *
 * If schema is not empty, types are taken from it instead of inferred, and keys not
 * in it are rejected, or ignored if ignore_unknown_keys. Ignored keys must still be
 * valid namelist syntax, since their values are parsed to find where they end. FLOAT
 * accepts integers and reals, INT only integers, BOOL only logicals, STR only quoted
 * strings, and INT_OR_STR integers or quoted strings, and the same for the elements of
 * VEC_* and VEC3_*. Each assignment of a key must give one value for scalars and 3 for
 * VEC3_* (null values count, e.g. "a = 1,,3").
 *
 * Keys found in aliases are deprecated. They are renamed to the key they map to, with
 * a warning printed to stderr, before the schema is checked, so the schema must use the
 * normalized keys. Using more than one name of the same key in a group (e.g.
 * "net_charge = 1, ml_charge = 1") is an error.
 * */
std::vector<NamelistGroup> parseNamelists(
    const std::string& text,
    const std::string& group_name = "",
    const NamelistSchema& schema = {},
    const NamelistAliases& aliases = {},
    bool ignore_unknown_keys = false
);

/**
 * Return the lowercased names of all namelist groups in the text, in order of
 * appearance, with repeats. The contents of the groups are not parsed, so they can't
 * cause errors.
 * */
std::vector<std::string> listNamelistGroupNames(const std::string& text);

}  // namespace fortran_utils
