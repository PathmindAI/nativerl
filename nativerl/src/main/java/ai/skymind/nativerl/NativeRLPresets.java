package ai.skymind.nativerl;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.javacpp.tools.*;

/**
 * This file basically contains the configuration to map the API from nativerl.h to Java using JavaCPP.
 */
@Properties(
    value = @Platform(
        compiler = "cpp11",
        define = "SHARED_PTR_NAMESPACE std",
        include = "nativerl.h"
    ),
    target = "ai.skymind.nativerl",
    global = "ai.skymind.nativerl.NativeRL"
)
public class NativeRLPresets implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("NATIVERL_EXPORT").annotations().cppTypes())
               .put(new Info("SSIZE_T").cast().valueTypes("long").pointerTypes("SizeTPointer"))
               .put(new Info("std::vector<float>").pointerTypes("FloatVector").define())
               .put(new Info("std::vector<ssize_t>").pointerTypes("SSizeTVector").define())
               .put(new Info("nativerl::Environment").virtualize());
    }
}
