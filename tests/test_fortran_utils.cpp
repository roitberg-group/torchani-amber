#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "fortran_utils.h"

using fortran_utils::namelist_out_of_range_error;
using fortran_utils::namelist_parse_error;
using fortran_utils::namelist_syntax_error;
using fortran_utils::namelist_unquoted_string_error;
using fortran_utils::namelist_unsupported_feature_error;
using fortran_utils::namelist_unterminated_group_error;
using fortran_utils::namelist_unterminated_string_error;
using fortran_utils::NamelistGroup;
using fortran_utils::NamelistValue;
using fortran_utils::parseNamelists;
using Values = std::vector<NamelistValue>;

static const NamelistValue null_value = std::monostate{};
static const std::filesystem::path namelists_dir =
    std::filesystem::path(__FILE__).parent_path() / "namelists";

static std::map<std::string, Values> parseSingleGroup(const std::string& text) {
    std::vector<NamelistGroup> groups = parseNamelists(text);
    REQUIRE(groups.size() == 1);
    return groups[0].values;
}

static std::string readFile(const std::filesystem::path& path) {
    std::ifstream file(path);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

TEST_CASE("Namelist null values", "[namelist]") {
    CHECK(parseSingleGroup("&g a= /").at("a") == Values{null_value});
    CHECK(parseSingleGroup("&g a=, /").at("a") == Values{null_value});
    CHECK(parseSingleGroup("&g a=1,,3 /").at("a") == Values{1L, null_value, 3L});
    CHECK(parseSingleGroup("&g a=,3 /").at("a") == Values{null_value, 3L});
    CHECK(parseSingleGroup("&g a=1, /").at("a") == Values{1L});
}

TEST_CASE("Namelist repeats and arrays", "[namelist]") {
    CHECK(parseSingleGroup("&g a=3*2 /").at("a") == Values{2L, 2L, 2L});
    CHECK(parseSingleGroup("&g a=2* , 5 /").at("a") == Values{null_value, null_value, 5L});
    CHECK(
        parseSingleGroup("&g a=2*'x' /").at("a")
        == Values{std::string{"x"}, std::string{"x"}}
    );
    CHECK(parseSingleGroup("&g a = 1 2 3 /").at("a") == Values{1L, 2L, 3L});
}

TEST_CASE("Namelist repeated keys", "[namelist]") {
    CHECK(parseSingleGroup("&g a=1,2,3 a=5 /").at("a") == Values{5L, 2L, 3L});
    CHECK(parseSingleGroup("&g a=1 a=, /").at("a") == Values{1L});
}

TEST_CASE("Namelist value types", "[namelist]") {
    auto values = parseSingleGroup(
        "&g a=t c=.true. d=true e=F h=1.0d-3 i=.5 j=5. k=1e3 l=-0 "
        "m=+3 n=1.0E+02 o='it''s' p=\"x/y!z\" /"
    );
    CHECK(values.at("a") == Values{true});
    CHECK(values.at("c") == Values{true});
    CHECK(values.at("d") == Values{true});
    CHECK(values.at("e") == Values{false});
    CHECK(values.at("h") == Values{1.0e-3});
    CHECK(values.at("i") == Values{0.5});
    CHECK(values.at("j") == Values{5.0});
    CHECK(values.at("k") == Values{1000.0});
    CHECK(values.at("l") == Values{0L});
    CHECK(values.at("m") == Values{3L});
    CHECK(values.at("n") == Values{100.0});
    CHECK(values.at("o") == Values{std::string{"it's"}});
    CHECK(values.at("p") == Values{std::string{"x/y!z"}});
}

TEST_CASE("Namelist group delimiters and comments", "[namelist]") {
    CHECK(parseSingleGroup("&g a=1 &end").at("a") == Values{1L});
    CHECK(parseSingleGroup("$g a=1 $end").at("a") == Values{1L});
    CHECK(parseSingleGroup("&g a=1/").at("a") == Values{1L});
    CHECK(parseSingleGroup("&G A=1 /").at("a") == Values{1L});
    CHECK(parseSingleGroup("&g\r\n a=1,\r\n b='x'\r\n/\r\n").at("b") == Values{std::string{"x"}});

    auto values = parseSingleGroup("&g\n a=1 ! comment / here\n b=2\n/");
    CHECK(values.at("a") == Values{1L});
    CHECK(values.at("b") == Values{2L});
}

TEST_CASE("Namelist multiple groups and text outside groups", "[namelist]") {
    std::vector<NamelistGroup> groups = parseNamelists(
        "title &x\n &g a=1 &end\n /\n&h b=2 / junk\n&g a=2 /\nDISANG=x\n&end\nRES 1 2\nEND\n"
    );
    REQUIRE(groups.size() == 3);
    CHECK(groups[0].name == "g");
    CHECK(groups[0].values.at("a") == Values{1L});
    CHECK(groups[1].name == "h");
    CHECK(groups[1].values.at("b") == Values{2L});
    CHECK(groups[2].name == "g");
    CHECK(groups[2].values.at("a") == Values{2L});
}

TEST_CASE("Namelist selected by name", "[namelist]") {
    std::string text = "&g a=1 /\n&h b=2 /\n&G a=3 /\n";
    std::vector<NamelistGroup> groups = parseNamelists(text, "G");
    REQUIRE(groups.size() == 2);
    CHECK(groups[0].name == "g");
    CHECK(groups[0].values.at("a") == Values{1L});
    CHECK(groups[1].name == "g");
    CHECK(groups[1].values.at("a") == Values{3L});

    CHECK(parseNamelists(text, "x").empty());
    CHECK(parseNamelists(text, "").size() == 3);

    std::string text_with_bad_group = "&bad a(1)=1, c=(1,2)\n&g a=1 /\n";
    CHECK_THROWS_AS(parseNamelists(text_with_bad_group), namelist_unsupported_feature_error);
    REQUIRE(parseNamelists(text_with_bad_group, "g").size() == 1);
}

TEST_CASE("Namelist unquoted strings", "[namelist]") {
    CHECK_THROWS_AS(parseNamelists("&g a=abc /"), namelist_unquoted_string_error);
    CHECK_THROWS_AS(parseNamelists("&g a=:1-10@CA /"), namelist_unquoted_string_error);
    CHECK_THROWS_AS(parseNamelists("&g a=.f /"), namelist_unquoted_string_error);
    CHECK_THROWS_AS(parseNamelists("&g a=.Fxyz /"), namelist_unquoted_string_error);
    CHECK_THROWS_AS(parseNamelists("&g a=3*x /"), namelist_unquoted_string_error);
    CHECK_THROWS_AS(parseNamelists("&g a=1, x /"), namelist_unquoted_string_error);
    CHECK_THROWS_AS(parseNamelists("&g a=0.1e /"), namelist_unquoted_string_error);
}

TEST_CASE("Namelist errors", "[namelist]") {
    CHECK_THROWS_AS(parseNamelists("&g a(1)=1 /"), namelist_unsupported_feature_error);
    CHECK_THROWS_AS(parseNamelists("&g a%b=1 /"), namelist_unsupported_feature_error);
    CHECK_THROWS_AS(parseNamelists("&g a=(1,2) /"), namelist_unsupported_feature_error);
    CHECK_THROWS_AS(parseNamelists("&g a=1"), namelist_unterminated_group_error);
    CHECK_THROWS_AS(parseNamelists("&g a='x /"), namelist_unterminated_string_error);
    CHECK_THROWS_AS(parseNamelists("&g a=1 &foo"), namelist_syntax_error);
    CHECK_THROWS_AS(parseNamelists("&g 1 /"), namelist_syntax_error);
    CHECK_THROWS_AS(parseNamelists("&g a 1 /"), namelist_syntax_error);
    CHECK_THROWS_AS(parseNamelists("&g a = = 1 /"), namelist_syntax_error);
    CHECK_THROWS_AS(parseNamelists("&g a=99999999999999999999 /"), namelist_out_of_range_error);
    CHECK_THROWS_AS(parseNamelists("&g a=1e400 /"), namelist_out_of_range_error);
    CHECK_THROWS_AS(
        parseNamelists("&g a=99999999999999999999*1 /"),
        namelist_out_of_range_error
    );
    CHECK_THROWS_AS(parseNamelists("&g a=abc /"), namelist_parse_error);
}

TEST_CASE("Namelist Amber inputs", "[namelist]") {
    REQUIRE(std::filesystem::is_directory(namelists_dir));
    int num_files = 0;
    for (const auto& entry : std::filesystem::directory_iterator(namelists_dir)) {
        INFO("File: " << entry.path().filename().string());
        std::vector<NamelistGroup> groups;
        REQUIRE_NOTHROW(groups = parseNamelists(readFile(entry.path())));
        bool file_has_cntrl = false;
        for (const NamelistGroup& group : groups) {
            file_has_cntrl = file_has_cntrl || group.name == "cntrl";
        }
        CHECK(file_has_cntrl);

        std::vector<NamelistGroup> cntrl_groups = parseNamelists(readFile(entry.path()), "cntrl");
        CHECK_FALSE(cntrl_groups.empty());
        for (const NamelistGroup& group : cntrl_groups) {
            CHECK(group.name == "cntrl");
        }
        ++num_files;
    }
    CHECK(num_files > 0);
}

TEST_CASE("Namelist Amber input values", "[namelist]") {
    SECTION("Text and '&end' after the namelists") {
        std::vector<NamelistGroup> groups =
            parseNamelists(readFile(namelists_dir / "examples__GBION__GBION_1BNA__min.in"));
        REQUIRE(groups.size() == 2);
        CHECK(groups[0].name == "cntrl");
        CHECK(groups[0].values.size() == 19);
        CHECK(groups[0].values.at("imin") == Values{1L});
        CHECK(groups[0].values.at("cut") == Values{9999.0});
        CHECK(groups[0].values.at("gi_coef_1_n") == Values{0.05});
        CHECK(groups[1].name == "wt");
        CHECK(groups[1].values.at("type") == Values{std::string{"END"}});
    }
    SECTION("'&end' followed by '/'") {
        std::vector<NamelistGroup> groups =
            parseNamelists(readFile(namelists_dir / "dat__SIRAH__tutorial__1__SANDER__eq_GB.in"));
        REQUIRE(groups.size() == 5);
        CHECK(groups[0].name == "cntrl");
        CHECK(groups[0].values.at("dt") == Values{0.02});
        CHECK(groups[0].values.at("ntr") == Values{0L});
        CHECK(groups[4].name == "wt");
        CHECK(groups[4].values.at("type") == Values{std::string{"END"}});
    }
}
