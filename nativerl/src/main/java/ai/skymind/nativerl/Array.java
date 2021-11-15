// Targeted by JavaCPP version 1.5.4: DO NOT EDIT THIS FILE

package ai.skymind.nativerl;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static ai.skymind.nativerl.NativeRL.*;


/**
 * A generic multidimensional array of 32-bit floating point elements with a very simple interface
 * such that it can be mapped and used easily with tools like JavaCPP and pybind11.
 */
@Namespace("nativerl") @NoOffset @Properties(inherit = ai.skymind.nativerl.NativeRLPresets.class)
public class Array extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Array(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public Array(long size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(long size);
    @Override public Array position(long position) {
        return (Array)super.position(position);
    }
    @Override public Array getPointer(long i) {
        return new Array(this).position(position + i);
    }

    public native FloatPointer allocated(); public native Array allocated(FloatPointer setter);
    public native FloatPointer data(); public native Array data(FloatPointer setter);
    public native @ByRef SSizeTVector shape(); public native Array shape(SSizeTVector setter);

    public Array() { super((Pointer)null); allocate(); }
    private native void allocate();
    public Array(@Const @ByRef Array a) { super((Pointer)null); allocate(a); }
    private native void allocate(@Const @ByRef Array a);
    public Array(FloatPointer data, @Const @ByRef SSizeTVector shape) { super((Pointer)null); allocate(data, shape); }
    private native void allocate(FloatPointer data, @Const @ByRef SSizeTVector shape);
    public Array(FloatBuffer data, @Const @ByRef SSizeTVector shape) { super((Pointer)null); allocate(data, shape); }
    private native void allocate(FloatBuffer data, @Const @ByRef SSizeTVector shape);
    public Array(float[] data, @Const @ByRef SSizeTVector shape) { super((Pointer)null); allocate(data, shape); }
    private native void allocate(float[] data, @Const @ByRef SSizeTVector shape);
    public Array(@Const @ByRef FloatVector values) { super((Pointer)null); allocate(values); }
    private native void allocate(@Const @ByRef FloatVector values);
    public Array(@Const @ByRef SSizeTVector shape) { super((Pointer)null); allocate(shape); }
    private native void allocate(@Const @ByRef SSizeTVector shape);
    public native @ByRef @Name("operator +=") Array addPut(@Const @ByRef Array a);

    public native @ByVal FloatVector values();

    public native @Cast("ssize_t") long length();

    public native @Cast("ssize_t") long py_len();

    public native float get_item(int i);

}
