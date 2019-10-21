// Targeted by JavaCPP version 1.5.1: DO NOT EDIT THIS FILE

package ai.skymind.nativerl;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static ai.skymind.nativerl.NativeRL.*;


@Namespace("nativerl") @Properties(inherit = ai.skymind.nativerl.NativeRLPresets.class)
public class Environment extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public Environment() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public Environment(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Environment(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public Environment position(long position) {
        return (Environment)super.position(position);
    }

    @Virtual(true) public native void setSeed(long seed);
    @Virtual(true) public native @Const Space getActionSpace();
    @Virtual(true) public native @Const Space getObservationSpace();
    @Virtual(true) public native @Const @ByRef Array getObservation();
    @Virtual(true) public native @Cast("bool") boolean isDone();
    @Virtual(true) public native void reset();
    @Virtual(true) public native float step(@Cast("ssize_t") long action);
    @Virtual(true) public native @Const @ByRef Array step(@Const @ByRef Array action);
}
