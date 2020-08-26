package ai.skymind.nativerl;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

/**
 * The factory method to create instances of arbitrary subclasses of Environment.
 * This gets exported to jniNativeRL.h as the C function createJavaEnvironment().
 * To release them we must call releaseJavaEnvironment().
 */
@Properties(inherit = ai.skymind.nativerl.NativeRLPresets.class)
public class CreateEnvironment extends FunctionPointer {

    static Map<Environment, Environment> instances = Collections.synchronizedMap(new HashMap<Environment, Environment>());

    public @Name("createJavaEnvironment") Environment call(String name) throws Exception {
        Environment e = Class.forName(name).asSubclass(Environment.class).newInstance();
        instances.put(e, e);
        return e;
    }
}
