#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>

/**
 * This file basically contains the configuration to map the API from nativerl.h to Python using pybind11.
 * Also provides init() and uninit() functions to load and unload the JVM via JavaCPP.
 */
PYBIND11_MAKE_OPAQUE(std::vector<float>);
PYBIND11_MAKE_OPAQUE(std::vector<ssize_t>);

#include "nativerl.h"
#include "jniNativeRL.h"

/** Loads the JVM using JavaCPP and the given arguments. */
int init(const std::vector<std::string>& jvmargs) {
    const char *argv[256];
    for (size_t i = 0; i < jvmargs.size() && i < 256; i++) {
        argv[i] = jvmargs[i].c_str();
    }
    return JavaCPP_init(jvmargs.size(), argv);
}

/** Unloads the JVM using JavaCPP. */
int uninit() {
    return JavaCPP_uninit();
}

PYBIND11_MODULE(nativerl, m) {
// Do not initialize here, let users pass arguments to the JVM via init()
//    JavaCPP_init(0, nullptr);

    pybind11::bind_vector<std::vector<float>>(m, "FloatVector", pybind11::buffer_protocol());
    pybind11::bind_vector<std::vector<ssize_t>>(m, "SSizeTVector", pybind11::buffer_protocol());

    pybind11::class_<nativerl::Array>(m, "Array", pybind11::buffer_protocol())
        .def(pybind11::init([](pybind11::buffer b) {
            pybind11::buffer_info info = b.request();

            if (info.format != pybind11::format_descriptor<float>::format()) {
                throw std::runtime_error("Incompatible format: expected a float array!");
            }

            return new nativerl::Array((float*)info.ptr, info.shape);
        }))
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
        .def_readwrite("n", &nativerl::Discrete::n)
        .def_readwrite("size", &nativerl::Discrete::size);

    pybind11::class_<nativerl::Environment>(m, "Environment")
        .def("getActionSpace", &nativerl::Environment::getActionSpace)
        .def("getActionMaskSpace", &nativerl::Environment::getActionMaskSpace)
        .def("getObservationSpace", &nativerl::Environment::getObservationSpace)
        .def("getActionMask", &nativerl::Environment::getActionMask)
        .def("getObservation", &nativerl::Environment::getObservation)
        .def("isDone", &nativerl::Environment::isDone)
        .def("reset", &nativerl::Environment::reset)
        .def("step", (float (nativerl::Environment::*)(const nativerl::Array& action))&nativerl::Environment::step)
//        .def("step", (const nativerl::Array& (nativerl::Environment::*)(const nativerl::Array& action))&nativerl::Environment::step);
        .def("getMetrics", &nativerl::Environment::getMetrics);

    m.def("createEnvironment", &createEnvironment);
    m.def("releaseEnvironment", &releaseEnvironment);

    m.def("init", &init);
    m.def("uninit", &uninit);
}
