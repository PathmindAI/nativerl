#!/bin/bash
set -eu

CXXSOURCES="nativerl.cpp"
JAVASOURCES="TrafficEnvironment.java"

CXXFLAGS="-O3 -Wall -shared -std=c++11 -fPIC -I. -Inativerl -L nativerl/linux-x86_64 -Wl,-rpath,nativerl/linux-x86_64 -ljniNativeRL"
JAVACPP="lib/javacpp.jar"
JAVAFLAGS="-cp .:$JAVACPP:lib/*:."

for I in $(java -jar $JAVACPP -print platform.includepath); do
    CXXFLAGS="$CXXFLAGS -I$I"
done
for L in $(java -jar $JAVACPP -print platform.linkpath); do
    CXXFLAGS="$CXXFLAGS -L$L -Wl,-rpath,$L"
done
javac $JAVAFLAGS nativerl/NativeRLPresets.java
java -jar $JAVACPP $JAVAFLAGS nativerl.NativeRLPresets
javac $JAVAFLAGS $JAVASOURCES nativerl/*.java
java -jar $JAVACPP $JAVAFLAGS 'nativerl.**' -header

g++ $CXXFLAGS `python3 -m pybind11 --includes` $CXXSOURCES -o nativerl`python3-config --extension-suffix`
