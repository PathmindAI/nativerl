// Targeted by JavaCPP version 1.5.4: DO NOT EDIT THIS FILE

package ai.skymind.nativerl;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static ai.skymind.nativerl.NativeRL.*;


/**
 * The pure virtual (abstract) interface of a "native" environment. This gets mapped,
 * for example, with JavaCPP and implemented by a Java class. The implementation needs
 * to export functions to create and release Environment objects. In the case of JavaCPP,
 * the createJavaEnvironment() and releaseJavaEnvironment() are available in the generated
 * jniNativeRL.h header file.
 * <p>
 * However, we can just as well implement it in pure C++, which we would do in the case of,
 * for example, ROS or MATLAB Simulink.
 * <p>
 * On the Python side, these functions are picked up by, for example, pybind11 and used
 * to implement Python interfaces of environments, such as gym.Env, for RLlib, etc.
 */
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
    @Override public Environment getPointer(long i) {
        return new Environment(this).position(position + i);
    }

    //    /** Passes a new random seed that should be used for reproducibility. */
    //    virtual void setSeed(long long seed) = 0;
    /** Returns the i-th action Space supported. */
    @Virtual(true) public native @Const Space getActionSpace(@Cast("ssize_t") long i/*=0*/);
    /** Returns the action mask Space supported. */
    @Virtual(true) public native @Const Space getActionMaskSpace();
    /** Returns the observation Space supported. */
    @Virtual(true) public native @Const Space getObservationSpace();
    /** Returns the metrics Space supported. */
    @Virtual(true) public native @Const Space getMetricsSpace();
    /** Returns the number of agents in this environment. */
    @Virtual(true) public native @Cast("ssize_t") long getNumberOfAgents();
    /** Returns the current state of the possible actions for the given agent. */
    @Virtual(true) public native @Const @ByRef Array getActionMask(@Cast("ssize_t") long agentId/*=0*/);
    /** Returns the current state of the simulation for the given agent. */
    @Virtual(true) public native @Const @ByRef Array getObservation(@Cast("ssize_t") long agentId/*=0*/);
    /** Indicates when the given agent is not available to have its state queried, do actions, etc. */
    @Virtual(true) public native @Cast("bool") boolean isSkip(@Cast("ssize_t") long agentId/*=-1*/);
    /** Indicates when a simulation episode is over for the given agent, or -1 for all. */
    @Virtual(true) public native @Cast("bool") boolean isDone(@Cast("ssize_t") long agentId/*=-1*/);
    /** Used to reset the simulation, preferably starting a new random sequence. */
    @Virtual(true) public native void reset();
    /** Sets the next action for the given agent to be done during the next step. */
    @Virtual(true) public native void setNextAction(@Const @ByRef Array action, @Cast("ssize_t") long agentId/*=0*/);
    /** Used to advance the simulation by a single step. */
    @Virtual(true) public native void step();
    /** Returns the reward based on variables for the given agent before and after the last step. */
    @Virtual(true) public native float getReward(@Cast("ssize_t") long agentId/*=0*/);
    /** Returns the last values of observationForReward() */
    @Virtual(true) public native @Const @ByRef Array getMetrics(@Cast("ssize_t") long agentId/*=0*/);
    /** Returns the reward terms */
    @Virtual(true) public native @Const @ByRef Array getRewardTerms(@Cast("ssize_t") long agentId/*=0*/);
}
