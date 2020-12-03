// Targeted by JavaCPP version 1.5.4: DO NOT EDIT THIS FILE

package ai.skymind.nativerl;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

public class NativeRL extends ai.skymind.nativerl.NativeRLPresets {
    static { Loader.load(); }

// Targeting FloatVector.java


// Targeting SSizeTVector.java


// Parsed from nativerl.h

// #ifndef NATIVERL_H
// #define NATIVERL_H

// #ifdef _WIN32
// #define NATIVERL_EXPORT __declspec(dllexport)
// #include <BaseTsd.h>
// #else
// #define NATIVERL_EXPORT __attribute__((visibility("default")))
// #endif

// #include <vector>

/**
 * This is the main C++ interface implemented, for example, in Java via JavaCPP,
 * and used in Python by, for example, RLlib via pybind11.
 */
// Targeting Array.java


// Targeting Space.java


// Targeting Continuous.java


// Targeting Discrete.java



/** Helper method to cast dynamically a Space object into Continuous. */

/** Helper method to cast dynamically a Space object into Discrete. */

// Targeting Environment.java



// #ifdef _WIN32
// Windows does not support undefined symbols in DLLs, disallowing circular dependencies,
// so we cannot call createEnvironment() defined in nativerl.cpp from Java...
@Namespace("nativerl") public static native @SharedPtr Environment createEnvironment(@Cast("const char*") BytePointer name);
@Namespace("nativerl") public static native @SharedPtr Environment createEnvironment(String name);
// #else
// #endif



// #endif // NATIVERL_H


}
