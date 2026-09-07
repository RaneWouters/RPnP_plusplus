cmake_minimum_required(VERSION 3.16)
project(r2ppnp_consumer LANGUAGES CXX)

find_package(r2ppnp 0.1 CONFIG REQUIRED)

add_executable(r2ppnp_consumer main.cpp)
target_link_libraries(r2ppnp_consumer PRIVATE r2ppnp::r2ppnp)

enable_testing()
add_test(NAME installed_consumer COMMAND r2ppnp_consumer)
