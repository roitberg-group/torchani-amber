#include "fortran_utils.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "utils.h"

namespace fortran_utils {

namespace {

long lineNumber(const std::string& text, std::size_t pos) {
    pos = std::min(pos, text.size());
    return std::count(text.begin(), text.begin() + pos, '\n') + 1;
}

template <typename Error>
[[noreturn]] void throwParseError(
    const std::string& text,
    std::size_t pos,
    const std::string& message
) {
    throw Error(
        "Namelist parse error at line " + std::to_string(lineNumber(text, pos)) + ": "
            + message
    );
}

bool isBlank(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

bool isNameChar(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

std::string toLower(std::string str) {
    std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    return str;
}

std::size_t nextLineStart(const std::string& text, std::size_t pos) {
    std::size_t newline = text.find('\n', pos);
    return newline == std::string::npos ? text.size() : newline + 1;
}

void skipBlanksAndComments(const std::string& text, std::size_t& pos) {
    while (pos < text.size()) {
        if (isBlank(text[pos])) {
            ++pos;
        } else if (text[pos] == '!') {
            pos = nextLineStart(text, pos);
        } else {
            return;
        }
    }
}

std::string readName(const std::string& text, std::size_t& pos) {
    std::size_t start = pos;
    while (pos < text.size() && isNameChar(text[pos])) {
        ++pos;
    }
    return text.substr(start, pos - start);
}

// A key is a name followed by "=", or by "(" or "%" (unsupported, but still keys)
bool keyStartsAt(const std::string& text, std::size_t pos) {
    if (pos >= text.size() || !std::isalpha(static_cast<unsigned char>(text[pos]))) {
        return false;
    }
    readName(text, pos);
    skipBlanksAndComments(text, pos);
    return pos < text.size() && (text[pos] == '=' || text[pos] == '(' || text[pos] == '%');
}

// Doubled delimiters are escaped delimiters. Strings may span lines (newlines are dropped)
std::string readQuoted(const std::string& text, std::size_t& pos) {
    std::size_t start = pos;
    char delim = text[pos];
    ++pos;
    std::string str;
    while (pos < text.size()) {
        char c = text[pos];
        ++pos;
        if (c == delim && pos < text.size() && text[pos] == delim) {
            str += delim;
            ++pos;
        } else if (c == delim) {
            return str;
        } else if (c != '\n' && c != '\r') {
            str += c;
        }
    }
    throwParseError<namelist_unterminated_string_error>(text, start, "unterminated string");
}

std::string readUnquoted(const std::string& text, std::size_t& pos) {
    std::size_t start = pos;
    while (pos < text.size()) {
        char c = text[pos];
        if (isBlank(c) || c == ',' || c == '/' || c == '!' || c == '=' || c == '\''
            || c == '"') {
            break;
        }
        ++pos;
    }
    return text.substr(start, pos - start);
}

std::string typeName(NamelistType type) {
    switch (type) {
        case NamelistType::FLOAT:
            return "FLOAT";
        case NamelistType::INT:
            return "INT";
        case NamelistType::STR:
            return "STR";
        case NamelistType::BOOL:
            return "BOOL";
        case NamelistType::INT_OR_STR:
            return "INT_OR_STR";
        case NamelistType::FLOAT_OR_STR:
            return "FLOAT_OR_STR";
        case NamelistType::VEC_FLOAT:
            return "VEC_FLOAT";
        case NamelistType::VEC_FLOAT_OR_STR:
            return "VEC_FLOAT_OR_STR";
        case NamelistType::VEC_INT:
            return "VEC_INT";
        case NamelistType::VEC_STR:
            return "VEC_STR";
        case NamelistType::VEC_BOOL:
            return "VEC_BOOL";
        case NamelistType::VEC3_FLOAT:
            return "VEC3_FLOAT";
        case NamelistType::VEC3_INT:
            return "VEC3_INT";
        case NamelistType::VEC3_STR:
            return "VEC3_STR";
        case NamelistType::VEC3_BOOL:
            return "VEC3_BOOL";
    }
    return "UNKNOWN";
}

NamelistType elementType(NamelistType type) {
    switch (type) {
        case NamelistType::VEC_FLOAT:
        case NamelistType::VEC3_FLOAT:
            return NamelistType::FLOAT;
        case NamelistType::VEC_FLOAT_OR_STR:
            return NamelistType::FLOAT_OR_STR;
        case NamelistType::VEC_INT:
        case NamelistType::VEC3_INT:
            return NamelistType::INT;
        case NamelistType::VEC_STR:
        case NamelistType::VEC3_STR:
            return NamelistType::STR;
        case NamelistType::VEC_BOOL:
        case NamelistType::VEC3_BOOL:
            return NamelistType::BOOL;
        default:
            return type;
    }
}

// Returns 0 if any number of values is accepted
std::size_t requiredNumValues(NamelistType type) {
    switch (type) {
        case NamelistType::VEC_FLOAT:
        case NamelistType::VEC_FLOAT_OR_STR:
        case NamelistType::VEC_INT:
        case NamelistType::VEC_STR:
        case NamelistType::VEC_BOOL:
            return 0;
        case NamelistType::VEC3_FLOAT:
        case NamelistType::VEC3_INT:
        case NamelistType::VEC3_STR:
        case NamelistType::VEC3_BOOL:
            return 3;
        default:
            return 1;
    }
}

bool isInt(const std::string& token) {
    static const std::regex int_regex{R"([+-]?\d+)"};
    return std::regex_match(token, int_regex);
}

// Integers also match, since "1" is a valid real
bool isReal(const std::string& token) {
    static const std::regex real_regex{R"([+-]?(\d+\.?\d*|\.\d+)([eEdD][+-]?\d+)?)"};
    return std::regex_match(token, real_regex);
}

bool isTrue(const std::string& token) {
    std::string lower = toLower(token);
    return lower == ".true." || lower == ".t." || lower == "true" || lower == "t";
}

bool isFalse(const std::string& token) {
    std::string lower = toLower(token);
    return lower == ".false." || lower == ".f." || lower == "false" || lower == "f";
}

// Only call with tokens that match isInt or isReal
NamelistValue convertNumber(
    const std::string& text,
    std::size_t token_start,
    const std::string& token,
    const std::string& key,
    NamelistType type
) {
    try {
        if (type == NamelistType::INT) {
            return std::stol(token);
        }
        std::string real = token;
        std::replace(real.begin(), real.end(), 'd', 'e');
        std::replace(real.begin(), real.end(), 'D', 'e');
        return std::stod(real);
    } catch (const std::out_of_range&) {
        throwParseError<namelist_out_of_range_error>(
            text,
            token_start,
            "value " + token + " for key " + key + " is out of range"
        );
    }
}

[[noreturn]] void throwUnquotedStringError(
    const std::string& text,
    std::size_t token_start,
    const std::string& token,
    const std::string& key
) {
    throwParseError<namelist_unquoted_string_error>(
        text,
        token_start,
        "unquoted value " + token + " for key " + key
            + " is not a number or a logical. Did you mean it to be a string?"
            + " If so, quote it: " + key + " = \"" + token + "\""
    );
}

// Strings must be quoted in namelist input, as gfortran requires
NamelistValue convertUnquoted(
    const std::string& text,
    std::size_t token_start,
    const std::string& token,
    const std::string& key
) {
    if (isInt(token)) {
        return convertNumber(text, token_start, token, key, NamelistType::INT);
    }
    if (isReal(token)) {
        return convertNumber(text, token_start, token, key, NamelistType::FLOAT);
    }
    if (isTrue(token)) {
        return true;
    }
    if (isFalse(token)) {
        return false;
    }
    throwUnquotedStringError(text, token_start, token, key);
}

NamelistValue convertUnquotedWithType(
    const std::string& text,
    std::size_t token_start,
    const std::string& token,
    const std::string& key,
    NamelistType type
) {
    bool type_is_int = type == NamelistType::INT || type == NamelistType::INT_OR_STR;
    if (type_is_int && isInt(token)) {
        return convertNumber(text, token_start, token, key, NamelistType::INT);
    }
    bool type_is_float = type == NamelistType::FLOAT || type == NamelistType::FLOAT_OR_STR;
    if (type_is_float && isReal(token)) {
        return convertNumber(text, token_start, token, key, NamelistType::FLOAT);
    }
    if (type == NamelistType::BOOL && isTrue(token)) {
        return true;
    }
    if (type == NamelistType::BOOL && isFalse(token)) {
        return false;
    }
    bool token_is_value = isReal(token) || isTrue(token) || isFalse(token);
    bool type_is_or_str =
        type == NamelistType::INT_OR_STR || type == NamelistType::FLOAT_OR_STR;
    if (type == NamelistType::STR || (type_is_or_str && !token_is_value)) {
        throwUnquotedStringError(text, token_start, token, key);
    }
    throwParseError<namelist_type_error>(
        text,
        token_start,
        "value " + token + " for key " + key + " is not of type " + typeName(type)
    );
}

// Values are separated by commas or blanks. A comma not preceded by a value
// (since the "=" or the previous comma) indicates a null value. "r*c" is "c" repeated
// "r" times, and "r*" is "r" null values. If element_type is empty, types are inferred
std::vector<NamelistValue> parseValues(
    const std::string& text,
    std::size_t& pos,
    const std::string& key,
    std::optional<NamelistType> element_type
) {
    static const std::regex repeat_regex{R"((\d+)\*(.*))"};
    std::vector<NamelistValue> values;
    bool comma_is_null = true;
    while (true) {
        skipBlanksAndComments(text, pos);
        if (pos >= text.size() || text[pos] == '/' || text[pos] == '&' || text[pos] == '$'
            || keyStartsAt(text, pos)) {
            break;
        }
        if (text[pos] == ',') {
            if (comma_is_null) {
                values.emplace_back(std::monostate{});
            }
            comma_is_null = true;
            ++pos;
            continue;
        }
        if (text[pos] == '(') {
            throwParseError<namelist_unsupported_feature_error>(
                text,
                pos,
                "complex values are not supported"
            );
        }
        std::size_t token_start = pos;
        std::string token = readUnquoted(text, pos);
        std::size_t num_repeats = 1;
        std::smatch repeat_match;
        bool is_repeat = std::regex_match(token, repeat_match, repeat_regex);
        if (is_repeat) {
            try {
                num_repeats = std::stoul(repeat_match[1].str());
            } catch (const std::out_of_range&) {
                throwParseError<namelist_out_of_range_error>(
                    text,
                    token_start,
                    "repeat count " + repeat_match[1].str() + " is out of range"
                );
            }
            token = repeat_match[2].str();
        }
        bool is_quoted = token.empty() && pos < text.size()
            && (text[pos] == '\'' || text[pos] == '"');
        bool type_accepts_str = element_type == NamelistType::STR
            || element_type == NamelistType::INT_OR_STR
            || element_type == NamelistType::FLOAT_OR_STR;
        NamelistValue value;
        if (is_quoted && element_type && !type_accepts_str) {
            throwParseError<namelist_type_error>(
                text,
                pos,
                "string value for key " + key + " is not of type " + typeName(*element_type)
            );
        } else if (is_quoted) {
            value = readQuoted(text, pos);
        } else if (!token.empty() && element_type) {
            value = convertUnquotedWithType(text, token_start, token, key, *element_type);
        } else if (!token.empty()) {
            value = convertUnquoted(text, token_start, token, key);
        } else if (!is_repeat) {
            throwParseError<namelist_syntax_error>(text, token_start, "expected a value");
        }
        values.insert(values.end(), num_repeats, value);
        comma_is_null = false;
    }
    if (values.empty()) {
        values.emplace_back(std::monostate{});
    }
    return values;
}

// Parse from just after "&name" up to the terminator, and leave pos at the start of
// the following line, since the rest of the line after the terminator is ignored
NamelistGroup parseGroup(
    const std::string& text,
    std::size_t& pos,
    std::string name,
    const NamelistSchema& schema,
    const NamelistAliases& aliases,
    bool ignore_unknown_keys
) {
    NamelistGroup group{std::move(name), {}};
    // Maps each normalized key to the name it was written with
    std::unordered_map<std::string, std::string> written_keys;
    while (true) {
        skipBlanksAndComments(text, pos);
        if (pos >= text.size()) {
            throwParseError<namelist_unterminated_group_error>(
                text,
                pos,
                "end of text inside group &" + group.name
            );
        }
        if (text[pos] == '/') {
            pos = nextLineStart(text, pos);
            return group;
        }
        if (text[pos] == '&' || text[pos] == '$') {
            std::size_t end_start = pos;
            ++pos;
            if (toLower(readName(text, pos)) != "end") {
                throwParseError<namelist_syntax_error>(text, end_start, "expected '&end' or '/'");
            }
            pos = nextLineStart(text, pos);
            return group;
        }
        if (text[pos] == ',') {
            ++pos;
            continue;
        }
        std::size_t key_start = pos;
        std::string written_key = toLower(readName(text, pos));
        if (written_key.empty()) {
            throwParseError<namelist_syntax_error>(text, key_start, "expected a key");
        }
        skipBlanksAndComments(text, pos);
        if (pos < text.size() && (text[pos] == '(' || text[pos] == '%')) {
            throwParseError<namelist_unsupported_feature_error>(
                text,
                pos,
                "indexed keys and derived types are not supported"
            );
        }
        if (pos >= text.size() || text[pos] != '=') {
            throwParseError<namelist_syntax_error>(
                text,
                pos,
                "expected '=' after key " + written_key
            );
        }
        ++pos;
        auto alias_it = aliases.find(written_key);
        std::string key = alias_it == aliases.end() ? written_key : alias_it->second;
        auto [written_key_it, key_is_new] = written_keys.try_emplace(key, written_key);
        if (!key_is_new && written_key_it->second != written_key) {
            throwParseError<namelist_alias_conflict_error>(
                text,
                key_start,
                "keys " + written_key_it->second + " and " + written_key
                    + " are names of the same key " + key + ", only one can be used"
            );
        }
        if (alias_it != aliases.end()) {
            utils::printWarning(
                "Namelist key " + written_key + " in group &" + group.name + " (line "
                + std::to_string(lineNumber(text, key_start)) + ") is deprecated."
                + " It is an alias of " + key + ", use " + key + " instead"
            );
        }
        std::optional<NamelistType> type;
        if (!schema.empty()) {
            auto type_it = schema.find(key);
            if (type_it == schema.end() && ignore_unknown_keys) {
                // Parsed with inferred types only to advance past its values
                parseValues(text, pos, written_key, std::nullopt);
                continue;
            }
            if (type_it == schema.end()) {
                throwParseError<namelist_unknown_key_error>(
                    text,
                    key_start,
                    "key " + written_key + " is not in the schema of group &" + group.name
                );
            }
            type = type_it->second;
        }
        std::optional<NamelistType> element_type;
        if (type) {
            element_type = elementType(*type);
        }
        std::vector<NamelistValue> new_values =
            parseValues(text, pos, written_key, element_type);
        std::size_t num_values = type ? requiredNumValues(*type) : 0;
        if (num_values != 0 && new_values.size() != num_values) {
            throwParseError<namelist_type_error>(
                text,
                key_start,
                "key " + written_key + " of type " + typeName(*type) + " takes "
                    + std::to_string(num_values) + " value(s), but "
                    + std::to_string(new_values.size()) + " were given"
            );
        }
        std::vector<NamelistValue>& values = group.values[key];
        if (values.size() < new_values.size()) {
            values.resize(new_values.size());
        }
        for (std::size_t i = 0; i != new_values.size(); ++i) {
            if (!std::holds_alternative<std::monostate>(new_values[i])) {
                values[i] = new_values[i];
            }
        }
    }
}

// If a group starts at the line that begins at pos, return its lowercased name, and set
// name_end to just after the name. Otherwise return an empty string
std::string groupNameAt(const std::string& text, std::size_t pos, std::size_t& name_end) {
    std::size_t token_start = text.find_first_not_of(" \t", pos);
    if (token_start == std::string::npos
        || (text[token_start] != '&' && text[token_start] != '$')) {
        return "";
    }
    name_end = token_start + 1;
    std::string name = toLower(readName(text, name_end));
    return name == "end" ? "" : name;
}

}  // namespace

std::vector<std::string> listNamelistGroupNames(const std::string& text) {
    std::vector<std::string> names;
    for (std::size_t pos = 0; pos < text.size(); pos = nextLineStart(text, pos)) {
        std::size_t name_end = 0;
        std::string name = groupNameAt(text, pos, name_end);
        if (!name.empty()) {
            names.push_back(name);
        }
    }
    return names;
}

std::vector<NamelistGroup> parseNamelists(
    const std::string& text,
    const std::string& group_name,
    const NamelistSchema& schema,
    const NamelistAliases& aliases,
    bool ignore_unknown_keys
) {
    NamelistSchema lower_schema;
    for (const auto& [key, type] : schema) {
        lower_schema[toLower(key)] = type;
    }
    NamelistAliases lower_aliases;
    for (const auto& [alias, key] : aliases) {
        lower_aliases[toLower(alias)] = toLower(key);
    }
    std::string selected_name = toLower(group_name);
    std::vector<NamelistGroup> groups;
    std::size_t pos = 0;
    while (pos < text.size()) {
        std::size_t name_end = 0;
        std::string name = groupNameAt(text, pos, name_end);
        bool group_is_selected = selected_name.empty() || name == selected_name;
        if (!name.empty() && group_is_selected) {
            pos = name_end;
            groups.push_back(parseGroup(text, pos, name, lower_schema, lower_aliases, ignore_unknown_keys));
            continue;
        }
        pos = nextLineStart(text, pos);
    }
    return groups;
}

}  // namespace fortran_utils
