NativeRL
========

Introduction
------------

This is a hack to allow using environments for reinforcement learning implemented in Java or C++, with Python frameworks such as RLlib or Intel Coach.

It defines an intermediary C++ interface via the classes in `nativerl.h`, which are mapped to Java using JavaCPP and the classes in the `nativerl` subdirectory. The `nativerl::Environment` class is meant to be subclassed by users to implement new environments, such as `TrafficEnvironment.java`. The C++ classes are then made available to Python via pybind11 as per the bindings defined in `nativerl.cpp`, which one can use afterwards to implement environments as part of Python APIs, for example, OpenAI Gym, as exemplified in `rllibtest.py`.


Required Software
-----------------

 * Linux or Mac (untested)
 * Clang or GCC
 * JDK 8+
 * JavaCPP 1.5.1+  https://github.com/bytedeco/javacpp
 * Python 3.7+  https://www.python.org/downloads/
 * pybind11 2.2.4+  https://github.com/pybind/pybind11


Build Instructions
------------------

 1. Install pybind11 and JavaCPP, the latter by running, for example, `mvn clean install` inside its source repository
 2. Place all required JAR files in the `lib` subdirectory, including `javacpp.jar` obtained from the previous step
 3. Modify the `CXXSOURCES` and `JAVASOURCES` variables at the top of `build.sh` as required
 4. Run `build.sh`
 5. Execute `rllibtest.py` or any other Python script to use the desired framework
