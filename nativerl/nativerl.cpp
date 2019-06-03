#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>

PYBIND11_MAKE_OPAQUE(std::vector<float>);
PYBIND11_MAKE_OPAQUE(std::vector<ssize_t>);

#include "nativerl.h"
#include "nativerl/jniNativeRL.h"

PYBIND11_MODULE(nativerl, m) {
    JavaCPP_init(0, nullptr);

    pybind11::bind_vector<std::vector<float>>(m, "FloatVector", pybind11::buffer_protocol());
    pybind11::bind_vector<std::vector<ssize_t>>(m, "SSizeTVector", pybind11::buffer_protocol());

    pybind11::class_<nativerl::Array>(m, "Array", pybind11::buffer_protocol())
        .def_buffer([](nativerl::Array &a) -> pybind11::buffer_info {
            std::vector<ssize_t> strides(a.shape.size());
            strides[a.shape.size() - 1] = sizeof(float);
            for (ssize_t i = a.shape.size() - 2; i >= 0; i--) {
                strides[i] = strides[i + 1] * a.shape[i + 1];
            }
            return pybind11::buffer_info(a.data, sizeof(float),
                pybind11::format_descriptor<float>::format(),
                a.shape.size(), a.shape, strides
            );
        });

    pybind11::class_<nativerl::Space>(m, "Space")
        .def("asContinuous", &nativerl::Space::asContinuous)
        .def("asDiscrete", &nativerl::Space::asDiscrete);

    pybind11::class_<nativerl::Continuous, nativerl::Space>(m, "Continuous")
        .def(pybind11::init<const std::vector<float>&,
                            const std::vector<float>&,
                            const std::vector<ssize_t>&>())
        .def_readwrite("low", &nativerl::Continuous::low)
        .def_readwrite("high", &nativerl::Continuous::high)
        .def_readwrite("shape", &nativerl::Continuous::shape);

    pybind11::class_<nativerl::Discrete, nativerl::Space>(m, "Discrete")
        .def(pybind11::init<ssize_t>())
        .def_readwrite("n", &nativerl::Discrete::n);

    pybind11::class_<nativerl::Environment>(m, "Environment")
        .def("getActionSpace", &nativerl::Environment::getActionSpace)
        .def("getObservationSpace", &nativerl::Environment::getObservationSpace)
        .def("getObservation", &nativerl::Environment::getObservation)
        .def("isDone", &nativerl::Environment::isDone)
        .def("reset", &nativerl::Environment::reset)
        .def("step", (float (nativerl::Environment::*)(ssize_t action))&nativerl::Environment::step)
        .def("step", (float (nativerl::Environment::*)(const nativerl::Array& action))&nativerl::Environment::step);

    m.def("createEnvironment", &createEnvironment);
    m.def("releaseEnvironment", &releaseEnvironment);
}
