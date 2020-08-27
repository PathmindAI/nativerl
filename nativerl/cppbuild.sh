#!/bin/bash
# Script to build the Python wrappers from nativerl.cpp with pybind11
set -eu

CXX='g++'
LINKFLAGS='-Wl,-rpath,$ORIGIN/'
if [[ $(uname -s) == "Darwin" ]]; then
    CXX="clang++ -undefined dynamic_lookup"
    LINKFLAGS="-Wl,-rpath,@loader_path/."
fi

# Indicate where to find the JNI library produced by JavaCPP, as required by the Python wrapper
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
$CXX $CXXFLAGS `python3-config --includes` `python3 -m pybind11 --includes` $CXXSOURCES -o $TARGETDIR/nativerl`python3-config --extension-suffix` $LINKFLAGS
