#include "fortran_utils.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace fortran_utils {

namespace {

template <typename Error>
[[noreturn]] void throwParseError(
    const std::string& text,
    std::size_t pos,
    const std::string& message
) {
    pos = std::min(pos, text.size());
    long line = std::count(text.begin(), text.begin() + pos, '\n') + 1;
    throw Error(
        "Namelist parse error at line " + std::to_string(line) + ": " + message
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

// Strings must be quoted in namelist input, as gfortran requires
NamelistValue convertUnquoted(
    const std::string& text,
    std::size_t token_start,
    const std::string& token,
    const std::string& key
) {
    static const std::regex int_regex{R"([+-]?\d+)"};
    static const std::regex real_regex{R"([+-]?(\d+\.?\d*|\.\d+)([eEdD][+-]?\d+)?)"};
    try {
        if (std::regex_match(token, int_regex)) {
            return std::stol(token);
        }
        if (std::regex_match(token, real_regex)) {
            std::string real = token;
            std::replace(real.begin(), real.end(), 'd', 'e');
            std::replace(real.begin(), real.end(), 'D', 'e');
            return std::stod(real);
        }
    } catch (const std::out_of_range&) {
        throwParseError<namelist_out_of_range_error>(
            text,
            token_start,
            "value " + token + " for key " + key + " is out of range"
        );
    }
    std::string lower = toLower(token);
    if (lower == ".true." || lower == ".t." || lower == "true" || lower == "t") {
        return true;
    }
    if (lower == ".false." || lower == ".f." || lower == "false" || lower == "f") {
        return false;
    }
    throwParseError<namelist_unquoted_string_error>(
        text,
        token_start,
        "unquoted value " + token + " for key " + key
            + " is not a number or a logical. Did you mean it to be a string?"
            + " If so, quote it: " + key + " = \"" + token + "\""
    );
}

// Values are separated by commas or blanks. A comma not preceded by a value
// (since the "=" or the previous comma) indicates a null value. "r*c" is "c" repeated
// "r" times, and "r*" is "r" null values
std::vector<NamelistValue> parseValues(
    const std::string& text,
    std::size_t& pos,
    const std::string& key
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
        NamelistValue value;
        if (is_quoted) {
            value = readQuoted(text, pos);
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
NamelistGroup parseGroup(const std::string& text, std::size_t& pos, std::string name) {
    NamelistGroup group{std::move(name), {}};
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
        std::string key = toLower(readName(text, pos));
        if (key.empty()) {
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
            throwParseError<namelist_syntax_error>(text, pos, "expected '=' after key " + key);
        }
        ++pos;
        std::vector<NamelistValue> new_values = parseValues(text, pos, key);
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

}  // namespace

std::vector<NamelistGroup> parseNamelists(
    const std::string& text,
    const std::string& group_name
) {
    std::string selected_name = toLower(group_name);
    std::vector<NamelistGroup> groups;
    std::size_t pos = 0;
    while (pos < text.size()) {
        std::size_t token_start = text.find_first_not_of(" \t", pos);
        if (token_start != std::string::npos
            && (text[token_start] == '&' || text[token_start] == '$')) {
            std::size_t name_end = token_start + 1;
            std::string name = toLower(readName(text, name_end));
            bool group_is_selected = selected_name.empty() || name == selected_name;
            if (!name.empty() && name != "end" && group_is_selected) {
                pos = name_end;
                groups.push_back(parseGroup(text, pos, name));
                continue;
            }
        }
        pos = nextLineStart(text, pos);
    }
    return groups;
}

}  // namespace fortran_utils
