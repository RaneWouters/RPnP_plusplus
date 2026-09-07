#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <Eigen/Core>
#include "r2ppnp/r2ppnp.h"
#include "r2ppnp/types.h"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "RPnP++: Hough Voting-based 2-Point RANSAC PnP";
    m.attr("__version__") = "0.1.0";

    py::class_<r2ppnp::Config>(m, "Config")
        .def(py::init<>())
        .def_readwrite("thv", &r2ppnp::Config::thv)
        .def_readwrite("nr1", &r2ppnp::Config::nr1)
        .def_readwrite("ransac_p", &r2ppnp::Config::ransac_p)
        .def_readwrite("max_trials", &r2ppnp::Config::max_trials)
        .def_readwrite("th_cons", &r2ppnp::Config::th_cons)
        .def_readwrite("thH_factor", &r2ppnp::Config::thH_factor)
        .def_readwrite("min_thH", &r2ppnp::Config::min_thH)
        .def_readwrite("thH_ratio", &r2ppnp::Config::thH_ratio)
        .def_readwrite("num_peaks", &r2ppnp::Config::num_peaks)
        .def_readwrite("gn_max_iter", &r2ppnp::Config::gn_max_iter)
        .def_readwrite("gn_converge", &r2ppnp::Config::gn_converge)
        .def_readwrite("seed", &r2ppnp::Config::seed)
        .def_readwrite("finalize", &r2ppnp::Config::finalize);

    py::class_<r2ppnp::PnPResult>(m, "PnPResult")
        .def(py::init<>())
        .def_readonly("success", &r2ppnp::PnPResult::success)
        .def_readonly("R", &r2ppnp::PnPResult::R)
        .def_readonly("t", &r2ppnp::PnPResult::t)
        .def_readonly("num_trials", &r2ppnp::PnPResult::num_trials)
        .def_readonly("num_inliers", &r2ppnp::PnPResult::num_inliers)
        .def_readonly("score", &r2ppnp::PnPResult::score)
        .def_readonly("inliers", &r2ppnp::PnPResult::inliers)
        .def_readonly("weights", &r2ppnp::PnPResult::weights)
        .def_readonly("errors", &r2ppnp::PnPResult::errors)
        .def_readonly("message", &r2ppnp::PnPResult::message);

    m.def("solve", &r2ppnp::r2ppnp_from_pixels,
        py::arg("world_points"), py::arg("pixel_points"),
        py::arg("K"), py::arg("th_pixel") = 10.0,
        py::arg("config") = r2ppnp::Config{},
        py::call_guard<py::gil_scoped_release>(),
        "Solve PnP from pixel coordinates");

    m.def("solve_normalized", &r2ppnp::r2ppnp,
        py::arg("world_points"), py::arg("norm_points"),
        py::arg("config") = r2ppnp::Config{},
        py::call_guard<py::gil_scoped_release>(),
        "Solve PnP from normalized image coordinates");

    m.def("refine", &r2ppnp::refine,
        py::arg("world_points"), py::arg("pixel_points"),
        py::arg("K"), py::arg("R_init"), py::arg("t_init"),
        py::arg("th_pixel") = 10.0,
        py::arg("config") = r2ppnp::Config{},
        py::call_guard<py::gil_scoped_release>(),
        "Refine an initial pose estimate");
}
