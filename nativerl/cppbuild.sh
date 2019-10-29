#!/bin/bash
# Script to build with pybind11 the Python wrappers from nativerl.cpp
set -eu

# Indicate where to find the JNI library produced by JavaCPP, as required by the Python wrappers
TARGETDIR="target"
PLATFORMDIR="$TARGETDIR/classes/ai/skymind/nativerl/$PLATFORM"

CXXSOURCES="$PLATFORMDIR/../nativerl.cpp"
CXXFLAGS="-O3 -Wall -shared -std=c++11 -fPIC -L$PLATFORMDIR/ -Wl,-rpath,$PLATFORMDIR/ -ljniNativeRL"

# Pick up the include and library directories provided by JavaCPP
PREVIFS="$IFS"
IFS="$PLATFORM_PATH_SEPARATOR"
for P in $PLATFORM_INCLUDEPATH; do
    CXXFLAGS="$CXXFLAGS -I$P/"
done
for P in $PLATFORM_LINKPATH; do
    CXXFLAGS="$CXXFLAGS -L$P/ -Wl,-rpath,$P/"
done
IFS="$PREVIFS"

# Compile the Python wrappers against the JNI library
cp $PLATFORMDIR/* $TARGETDIR/
g++ $CXXFLAGS `python3 -m pybind11 --includes` $CXXSOURCES -o $TARGETDIR/nativerl`python3-config --extension-suffix` '-Wl,-rpath,$ORIGIN/'
