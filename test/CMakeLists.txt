#TODO: This is not well integrated, look into CTest and Catch2 interface with Cmake
message(STATUS "Constructing Makefile for unit tests in ${CMAKE_CURRENT_BINARY_DIR}")
execute_process(COMMAND python ${CMAKE_CURRENT_LIST_DIR}/generate_test_values.py)
file(GLOB UNIT_TESTS_SOURCES CONFIGURE_DEPENDS "*.cpp")

add_executable(unit_tests ${UNIT_TESTS_SOURCES})
set_target_properties(unit_tests PROPERTIES CXX_STANDARD 14)
target_link_libraries(unit_tests PRIVATE torchani)
target_include_directories(unit_tests PRIVATE ../include ../external/catch2/include)
