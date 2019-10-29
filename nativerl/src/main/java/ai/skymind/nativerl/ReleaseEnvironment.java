package ai.skymind.nativerl;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

/**
 * The factory method to release instances of arbitrary subclasses of Environment.
 * This gets exported to jniNativeRL.h as the C function releaseEnvironment().
 * This must be used to release objects created with createEnvironment().
 */
@Properties(inherit = ai.skymind.nativerl.NativeRLPresets.class)
public class ReleaseEnvironment extends FunctionPointer {
    /** @param environment the instance of a sublass of Environment to release */
    public @Name("releaseEnvironment") void call(Environment environment) throws Exception {
        Environment e = CreateEnvironment.instances.remove(new Pointer(environment));
        if (e != null) {
            e.deallocate();
        }
    }
}
