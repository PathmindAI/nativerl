package ai.skymind.nativerl;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

/**
 * The factory method to release instances of arbitrary subclasses of Environment.
 * This gets exported to jniNativeRL.h as the C function releaseJavaEnvironment().
 * This must be used to release objects created with createJavaEnvironment().
 */
@Properties(inherit = ai.skymind.nativerl.NativeRLPresets.class)
public class ReleaseEnvironment extends FunctionPointer {
    public @Name("releaseJavaEnvironment") void call(Environment environment) throws Exception {
        Environment e = CreateEnvironment.instances.remove(new Pointer(environment));
        if (e != null) {
            e.deallocate();
        }
    }
}
