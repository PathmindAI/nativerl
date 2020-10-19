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

namespace nativerl {

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

class PythonEnvironment : public Environment {
public:
    /* Inherit the constructors */
    using Environment::Environment;

    const Space* getActionSpace(ssize_t i = 0) override {
        PYBIND11_OVERLOAD_PURE(const Space*, Environment, getActionSpace, i);
    }
    const Space* getActionMaskSpace() override {
        PYBIND11_OVERLOAD_PURE(const Space*, Environment, getActionMaskSpace);
    }
    const Space* getObservationSpace() override {
        PYBIND11_OVERLOAD_PURE(const Space*, Environment, getObservationSpace);
    }
    ssize_t getNumberOfAgents() override {
        PYBIND11_OVERLOAD_PURE(ssize_t, Environment, getNumberOfAgents);
    }
    ssize_t getRewardVariableCount() override {
        PYBIND11_OVERLOAD_PURE(ssize_t, Environment, getRewardVariableCount);
    }
    const Array& getActionMask(ssize_t agentId = 0) override {
        PYBIND11_OVERLOAD_PURE(const Array&, Environment, getActionMask, agentId);
    }
    const Array& getObservation(ssize_t agentId = 0) override {
        PYBIND11_OVERLOAD_PURE(const Array&, Environment, getObservation, agentId);
    }
    bool isSkip(ssize_t agentId = 0) override {
        PYBIND11_OVERLOAD_PURE(bool, Environment, isSkip, agentId);
    }
    bool isDone(ssize_t agentId = -1) override {
        PYBIND11_OVERLOAD_PURE(bool, Environment, isDone, agentId);
    }
    void reset() override {
        PYBIND11_OVERLOAD_PURE(void, Environment, reset);
    }
    void setNextAction(const Array& action, ssize_t agentId = 0) override {
        PYBIND11_OVERLOAD_PURE(void, Environment, setNextAction, action, agentId);
    }
    void step() override {
        PYBIND11_OVERLOAD_PURE(void, Environment, step);
    }
    float getReward(ssize_t agentId = 0) override {
        PYBIND11_OVERLOAD_PURE(float, Environment, getReward, agentId);
    }
    const Array& getMetrics(ssize_t agentId = 0) override {
        PYBIND11_OVERLOAD_PURE(const Array&, Environment, getMetrics, agentId);
    }
};

std::shared_ptr<Environment> createEnvironment(const char* name) {
    try {
        return std::shared_ptr<Environment>(createJavaEnvironment(name),
                            [](Environment *e) { releaseJavaEnvironment(e); });
    } catch (const std::exception &e) {
        // probably not a Java environment...
        std::cerr << "Warning: " << e.what() << std::endl;

        std::string s(name);
        size_t n = s.rfind('.');
        pybind11::object mod = pybind11::module::import(s.substr(0, n).c_str());
        pybind11::object cls = mod.attr(s.substr(n + 1).c_str());
        pybind11::object obj = cls();
        PythonEnvironment* ptr = obj.cast<PythonEnvironment*>();
        std::shared_ptr<pybind11::object> ref = std::make_shared<pybind11::object>(obj);
        return std::shared_ptr<Environment>(ref, ptr);
    }
}

}

PYBIND11_MODULE(nativerl, m) {
// Do not initialize here, let users pass arguments to the JVM via init()
//    JavaCPP_init(0, nullptr);

    pybind11::bind_vector<std::vector<float>>(m, "FloatVector", pybind11::buffer_protocol());
    pybind11::bind_vector<std::vector<ssize_t>>(m, "SSizeTVector", pybind11::buffer_protocol());

    pybind11::class_<nativerl::Array>(m, "Array", pybind11::buffer_protocol())
        .def(pybind11::init<const nativerl::Array &>())
        .def(pybind11::init<const std::vector<float>&>())
        .def(pybind11::init<const std::vector<ssize_t>&>())
        .def("values", &nativerl::Array::values)
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

    pybind11::class_<nativerl::Environment, nativerl::PythonEnvironment, std::shared_ptr<nativerl::Environment>>(m, "Environment")
        .def(pybind11::init<>())
        .def("getActionSpace", &nativerl::Environment::getActionSpace, pybind11::arg("i") = 0)
        .def("getActionMaskSpace", &nativerl::Environment::getActionMaskSpace)
        .def("getObservationSpace", &nativerl::Environment::getObservationSpace)
        .def("getNumberOfAgents", &nativerl::Environment::getNumberOfAgents)
        .def("getRewardVariableCount", &nativerl::Environment::getRewardVariableCount)
        .def("getActionMask", &nativerl::Environment::getActionMask, pybind11::arg("agentId") = 0)
        .def("getObservation", &nativerl::Environment::getObservation, pybind11::arg("agentId") = 0)
        .def("isSkip", &nativerl::Environment::isSkip, pybind11::arg("agentId") = 0)
        .def("isDone", &nativerl::Environment::isDone, pybind11::arg("agentId") = -1)
        .def("reset", &nativerl::Environment::reset)
        .def("setNextAction", &nativerl::Environment::setNextAction, pybind11::arg("action"), pybind11::arg("agentId") = 0)
        .def("step", &nativerl::Environment::step)
        .def("getReward", &nativerl::Environment::getReward, pybind11::arg("agentId") = 0)
        .def("getMetrics", &nativerl::Environment::getMetrics, pybind11::arg("agentId") = 0);

    m.def("createEnvironment", &nativerl::createEnvironment);

    m.def("init", &nativerl::init);
    m.def("uninit", &nativerl::uninit);
}
