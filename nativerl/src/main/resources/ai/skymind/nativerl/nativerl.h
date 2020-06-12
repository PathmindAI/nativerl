#ifndef NATIVERL_H
#define NATIVERL_H

#include <vector>

namespace nativerl {

class Array {
public:
    float* allocated;
    float* data;
    std::vector<ssize_t> shape;

    Array() : data(nullptr) { }
    Array(const Array &a)
            : allocated(nullptr), data(a.data), shape(a.shape) { }
    Array(float *data, const std::vector<ssize_t>& shape)
            : allocated(nullptr), data(data), shape(shape) { }
    Array(const std::vector<ssize_t>& shape)
            : allocated(nullptr), data(nullptr), shape(shape) {
        allocated = data = new float[length()];
    }
    ~Array() {
        delete[] allocated;
    }

    ssize_t length() {
        ssize_t length = 1;
        for (size_t i = 0; i < shape.size(); i++) {
            length *= shape[i];
        }
        return length;
    }
};

class Continuous;
class Discrete;

class Space {
public:
    virtual ~Space() { };
    Continuous* asContinuous();
    Discrete* asDiscrete();
};

class Continuous : public Space {
public:
    std::vector<float> low;
    std::vector<float> high;
    std::vector<ssize_t> shape;

    Continuous(const std::vector<float>& low,
               const std::vector<float>& high,
               const std::vector<ssize_t>& shape)
            : low(low), high(high), shape(shape) { }
};

class Discrete : public Space {
public:
    ssize_t n;

    Discrete(const Discrete& d) : n(d.n) { }
    Discrete(ssize_t n) : n(n) { }
};

Continuous* Space::asContinuous() { return dynamic_cast<Continuous*>(this); }
Discrete* Space::asDiscrete() { return dynamic_cast<Discrete*>(this); }

class Environment {
public:
    virtual ~Environment() { };
    virtual const Space* getActionSpace() = 0;
    virtual const Space* getObservationSpace() = 0;
    virtual const Array& getObservation() = 0;
    virtual bool isDone() = 0;
    virtual void reset() = 0;
    virtual float step(ssize_t action) = 0;
    virtual const Array& step(const Array& action) = 0;
    virtual const Array& getMetrics() = 0;
};

// typedef Environment* (*CreateEnvironment)(const char* name);
// typedef void (*ReleaseEnvironment)(Environment* environment);

}

#endif // NATIVERL_H
