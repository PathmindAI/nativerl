#!/bin/bash
set -eu

TARGETDIR="target"
PLATFORMDIR="$TARGETDIR/classes/ai/skymind/nativerl/$PLATFORM"

CXXSOURCES="$PLATFORMDIR/../nativerl.cpp"
CXXFLAGS="-O3 -Wall -shared -std=c++11 -fPIC -L$PLATFORMDIR/ -Wl,-rpath,$PLATFORMDIR/ -ljniNativeRL"

PREVIFS="$IFS"
IFS="$PLATFORM_PATH_SEPARATOR"
for P in $PLATFORM_INCLUDEPATH; do
    CXXFLAGS="$CXXFLAGS -I$P/"
done
for P in $PLATFORM_LINKPATH; do
    CXXFLAGS="$CXXFLAGS -L$P/ -Wl,-rpath,$P/"
done
IFS="$PREVIFS"

cp $PLATFORMDIR/* $TARGETDIR/
g++ $CXXFLAGS `python3 -m pybind11 --includes` $CXXSOURCES -o $TARGETDIR/nativerl`python3-config --extension-suffix` '-Wl,-rpath,$ORIGIN/'
