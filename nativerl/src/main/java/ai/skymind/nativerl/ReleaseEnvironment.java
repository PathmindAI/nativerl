package ai.skymind.nativerl;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Properties(inherit = ai.skymind.nativerl.NativeRLPresets.class)
public class ReleaseEnvironment extends FunctionPointer {
    public @Name("releaseEnvironment") void call(Environment environment) throws Exception {
        Environment e = CreateEnvironment.instances.remove(new Pointer(environment));
        if (e != null) {
            e.deallocate();
        }
    }
}
