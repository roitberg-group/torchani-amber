#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "fortran_utils.h"

using fortran_utils::namelist_alias_conflict_error;
using fortran_utils::namelist_out_of_range_error;
using fortran_utils::namelist_parse_error;
using fortran_utils::namelist_syntax_error;
using fortran_utils::namelist_type_error;
using fortran_utils::namelist_unknown_key_error;
using fortran_utils::namelist_unquoted_string_error;
using fortran_utils::namelist_unsupported_feature_error;
using fortran_utils::namelist_unterminated_group_error;
using fortran_utils::namelist_unterminated_string_error;
using fortran_utils::NamelistAliases;
using fortran_utils::NamelistGroup;
using fortran_utils::NamelistSchema;
using fortran_utils::NamelistType;
using fortran_utils::NamelistValue;
using fortran_utils::listNamelistGroupNames;
using fortran_utils::parseNamelists;
using Values = std::vector<NamelistValue>;

static const NamelistValue null_value = std::monostate{};
static const std::filesystem::path namelists_dir =
    std::filesystem::path(__FILE__).parent_path() / "namelists";

static std::map<std::string, Values> parseSingleGroup(
    const std::string& text,
    const NamelistSchema& schema = {}
) {
    std::vector<NamelistGroup> groups = parseNamelists(text, "", schema);
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

TEST_CASE("Namelist schema scalars", "[namelist]") {
    NamelistSchema schema{
        {"f", NamelistType::FLOAT},
        {"i", NamelistType::INT},
        {"s", NamelistType::STR},
        {"b", NamelistType::BOOL},
    };
    auto values = parseSingleGroup("&g f=1 i=2 s='x' b=t /", schema);
    CHECK(values.at("f") == Values{1.0});
    CHECK(values.at("i") == Values{2L});
    CHECK(values.at("s") == Values{std::string{"x"}});
    CHECK(values.at("b") == Values{true});

    CHECK(parseSingleGroup("&g f=1.5d0 /", schema).at("f") == Values{1.5});
    CHECK(parseSingleGroup("&g b=.FALSE. /", schema).at("b") == Values{false});
    CHECK(parseSingleGroup("&g i= /", schema).at("i") == Values{null_value});
    CHECK(parseSingleGroup("&g I=2 /", {{"I", NamelistType::INT}}).at("i") == Values{2L});

    CHECK_THROWS_AS(parseNamelists("&g i=1.00 /", "", schema), namelist_type_error);
    CHECK_THROWS_AS(parseNamelists("&g i=t /", "", schema), namelist_type_error);
    CHECK_THROWS_AS(parseNamelists("&g i='1' /", "", schema), namelist_type_error);
    CHECK_THROWS_AS(parseNamelists("&g f=t /", "", schema), namelist_type_error);
    CHECK_THROWS_AS(parseNamelists("&g b=1 /", "", schema), namelist_type_error);
    CHECK_THROWS_AS(parseNamelists("&g b='t' /", "", schema), namelist_type_error);
    CHECK_THROWS_AS(parseNamelists("&g s=x /", "", schema), namelist_unquoted_string_error);
    CHECK_THROWS_AS(parseNamelists("&g s=1 /", "", schema), namelist_unquoted_string_error);
    CHECK_THROWS_AS(parseNamelists("&g i=1,2 /", "", schema), namelist_type_error);
    CHECK_THROWS_AS(parseNamelists("&g i=2*1 /", "", schema), namelist_type_error);
    CHECK_THROWS_AS(parseNamelists("&g x=1 /", "", schema), namelist_unknown_key_error);
    CHECK_THROWS_AS(parseNamelists("&g i=1e400 /", "", schema), namelist_type_error);
    CHECK_THROWS_AS(parseNamelists("&g f=1e400 /", "", schema), namelist_out_of_range_error);
}

TEST_CASE("Namelist schema with ignored unknown keys", "[namelist]") {
    NamelistSchema schema{{"i", NamelistType::INT}, {"f", NamelistType::FLOAT}};
    std::vector<NamelistGroup> groups =
        parseNamelists("&g x = 'a', 3*2, i = 1, y = .t., f = 2 /", "", schema, {}, true);
    REQUIRE(groups.size() == 1);
    CHECK(groups[0].values.size() == 2);
    CHECK(groups[0].values.at("i") == Values{1L});
    CHECK(groups[0].values.at("f") == Values{2.0});

    CHECK_THROWS_AS(
        parseNamelists("&g i = 1.5, x = 1 /", "", schema, {}, true),
        namelist_type_error
    );
    CHECK_THROWS_AS(
        parseNamelists("&g x = abc /", "", schema, {}, true),
        namelist_unquoted_string_error
    );
}

TEST_CASE("Namelist schema vectors", "[namelist]") {
    NamelistSchema schema{
        {"vf", NamelistType::VEC_FLOAT},
        {"vi", NamelistType::VEC_INT},
        {"vs", NamelistType::VEC_STR},
        {"vb", NamelistType::VEC_BOOL},
        {"v3f", NamelistType::VEC3_FLOAT},
        {"v3i", NamelistType::VEC3_INT},
        {"v3s", NamelistType::VEC3_STR},
        {"v3b", NamelistType::VEC3_BOOL},
    };
    auto values = parseSingleGroup(
        "&g vf=1,2.5 vi=3*4 vs='a' 'b' vb=t,.false. v3f=1 0 0 v3i=1,,3 "
        "v3s='x','y','z' v3b=2*f,t /",
        schema
    );
    CHECK(values.at("vf") == Values{1.0, 2.5});
    CHECK(values.at("vi") == Values{4L, 4L, 4L});
    CHECK(values.at("vs") == Values{std::string{"a"}, std::string{"b"}});
    CHECK(values.at("vb") == Values{true, false});
    CHECK(values.at("v3f") == Values{1.0, 0.0, 0.0});
    CHECK(values.at("v3i") == Values{1L, null_value, 3L});
    CHECK(values.at("v3s") == Values{std::string{"x"}, std::string{"y"}, std::string{"z"}});
    CHECK(values.at("v3b") == Values{false, false, true});

    CHECK(parseSingleGroup("&g vi=1 /", schema).at("vi") == Values{1L});
    CHECK(parseSingleGroup("&g v3i=3*7 /", schema).at("v3i") == Values{7L, 7L, 7L});

    CHECK_THROWS_AS(parseNamelists("&g vi=1,2.0 /", "", schema), namelist_type_error);
    CHECK_THROWS_AS(parseNamelists("&g vs='a',b /", "", schema), namelist_unquoted_string_error);
    CHECK_THROWS_AS(parseNamelists("&g v3f=1,2 /", "", schema), namelist_type_error);
    CHECK_THROWS_AS(parseNamelists("&g v3f=1,2,3,4 /", "", schema), namelist_type_error);
    CHECK_THROWS_AS(parseNamelists("&g v3i=1,2,t /", "", schema), namelist_type_error);
    CHECK_THROWS_AS(parseNamelists("&g vb=t,1 /", "", schema), namelist_type_error);
    CHECK_THROWS_AS(parseNamelists("&g v3s='x','y' /", "", schema), namelist_type_error);
    CHECK_THROWS_AS(parseNamelists("&g v3s='x','y',z /", "", schema), namelist_unquoted_string_error);
    CHECK_THROWS_AS(parseNamelists("&g v3b=t,f,'t' /", "", schema), namelist_type_error);
}

TEST_CASE("Namelist schema int or str", "[namelist]") {
    NamelistSchema schema{{"k", NamelistType::INT_OR_STR}};
    CHECK(parseSingleGroup("&g k = 2 /", schema).at("k") == Values{2L});
    CHECK(parseSingleGroup("&g k = 'Abc' /", schema).at("k") == Values{std::string{"Abc"}});
    CHECK_THROWS_AS(parseNamelists("&g k = abc /", "", schema), namelist_unquoted_string_error);
    CHECK_THROWS_AS(parseNamelists("&g k = 1.5 /", "", schema), namelist_type_error);
    CHECK_THROWS_AS(parseNamelists("&g k = t /", "", schema), namelist_type_error);
    CHECK_THROWS_AS(parseNamelists("&g k = 1, 2 /", "", schema), namelist_type_error);
}

TEST_CASE("Namelist schema float or str", "[namelist]") {
    NamelistSchema schema{
        {"k", NamelistType::FLOAT_OR_STR},
        {"vk", NamelistType::VEC_FLOAT_OR_STR},
    };
    CHECK(parseSingleGroup("&g k = 2 /", schema).at("k") == Values{2.0});
    CHECK(parseSingleGroup("&g k = 1.5 /", schema).at("k") == Values{1.5});
    CHECK(parseSingleGroup("&g k = 'Abc' /", schema).at("k") == Values{std::string{"Abc"}});
    CHECK(parseSingleGroup("&g vk = 1, , 2.5 /", schema).at("vk") == Values{1.0, null_value, 2.5});
    CHECK(parseSingleGroup("&g vk = 'a' /", schema).at("vk") == Values{std::string{"a"}});
    CHECK_THROWS_AS(parseNamelists("&g k = abc /", "", schema), namelist_unquoted_string_error);
    CHECK_THROWS_AS(parseNamelists("&g k = t /", "", schema), namelist_type_error);
    CHECK_THROWS_AS(parseNamelists("&g k = 1, 2 /", "", schema), namelist_type_error);
    CHECK_THROWS_AS(parseNamelists("&g vk = 1, t /", "", schema), namelist_type_error);
}

TEST_CASE("Namelist aliases", "[namelist]") {
    NamelistAliases aliases{{"old", "new"}, {"Old2", "NEW"}};
    std::vector<NamelistGroup> groups = parseNamelists("&g old=1 x=2 /", "", {}, aliases);
    REQUIRE(groups.size() == 1);
    CHECK(groups[0].values.at("new") == Values{1L});
    CHECK(groups[0].values.at("x") == Values{2L});
    CHECK(groups[0].values.count("old") == 0);

    groups = parseNamelists("&g OLD2=1,2 old2=3 /", "", {}, aliases);
    REQUIRE(groups.size() == 1);
    CHECK(groups[0].values.at("new") == Values{3L, 2L});

    NamelistSchema schema{{"new", NamelistType::FLOAT}};
    groups = parseNamelists("&g old=1 /", "", schema, aliases);
    REQUIRE(groups.size() == 1);
    CHECK(groups[0].values.at("new") == Values{1.0});

    CHECK_THROWS_AS(
        parseNamelists("&g old=1 new=2 /", "", {}, aliases),
        namelist_alias_conflict_error
    );
    CHECK_THROWS_AS(
        parseNamelists("&g old=1 old2=2 /", "", {}, aliases),
        namelist_alias_conflict_error
    );
    CHECK_THROWS_AS(
        parseNamelists("&g old=1 /", "", {{"old", NamelistType::INT}}, aliases),
        namelist_unknown_key_error
    );
    CHECK_NOTHROW(parseNamelists("&g old=1 /\n&g new=2 /", "", {}, aliases));
}

TEST_CASE("Namelist aliases print a deprecation warning", "[namelist]") {
    NamelistAliases aliases{{"old", "new"}};
    std::stringstream captured;
    std::streambuf* cerr_buf = std::cerr.rdbuf(captured.rdbuf());
    parseNamelists("&g\n new=1 /\n&g\n\n old=2 /\n", "", {}, aliases);
    std::cerr.rdbuf(cerr_buf);
    std::string warning = captured.str();
    CHECK(warning.find("WARNING (libtorchani)") != std::string::npos);
    CHECK(warning.find("key old in group &g (line 5) is deprecated") != std::string::npos);
    CHECK(warning.find("use new instead") != std::string::npos);
    CHECK(warning.find("WARNING", 1) == std::string::npos);
}

TEST_CASE("Namelist group names", "[namelist]") {
    std::string text = "title &x\n &G a=(1,2) &end\n /\n$h b=2 / junk\n&g a=2 /\n&end\n";
    CHECK(listNamelistGroupNames(text) == std::vector<std::string>{"g", "h", "g"});
    CHECK(listNamelistGroupNames("no groups\n").empty());
    CHECK(listNamelistGroupNames("").empty());
}
