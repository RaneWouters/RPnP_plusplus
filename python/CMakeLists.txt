find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core MODULE bindings.cpp)
target_link_libraries(_core PRIVATE r2ppnp)
set_target_properties(_core PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
install(TARGETS _core DESTINATION rpnp_pp)
