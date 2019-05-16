NativeRL
========

Introduction
------------

This is a hack to allow using environments for reinforcement learning implemented in Java or C++, with Python frameworks such as RLlib or Intel Coach.

Required Software
-----------------

 * Linux or Mac (untested)
 * Clang or GCC
 * JDK 8+
 * JavaCPP 1.5+
 * Python 3+
 * pybind11 2.2+

Build Instructions
------------------

 1. Place all required JAR files in the `lib` subdirectory, including `javacpp.jar`
 2. Modify the `CXXSOURCES` and `JAVASOURCES` variables at the top of `build.sh` as required
 3. Run `build.sh`
 4. Execute `rllibtest.py` or any other Python script to use the desired framework
