package nativerl;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.javacpp.tools.*;

@Properties(
    value = @Platform(
        include = "../nativerl.h"
    ),
    target = "nativerl",
    global = "nativerl.NativeRL"
)
public class NativeRLPresets implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("std::vector<float>").pointerTypes("FloatVector").define())
               .put(new Info("std::vector<ssize_t>").pointerTypes("SSizeTVector").define())
               .put(new Info("nativerl::Environment").virtualize());
    }
}
