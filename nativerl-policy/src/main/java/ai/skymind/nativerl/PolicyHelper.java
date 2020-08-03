package ai.skymind.nativerl;

/**
 * This is an interface that needs to be implemented by helper classes
 * that help users execute already trained reinforcement learning policies.
 * We can disable it at runtime by setting the "ai.skymind.nativerl.disablePolicyHelper"
 * system property to true, for example, during training.
 */
public interface PolicyHelper {
    static final boolean disablePolicyHelper = Boolean.getBoolean("ai.skymind.nativerl.disablePolicyHelper");

    /** Adapter from float to double array for {@link #computeActions(float[])}. */
    default public double[] computeActions(double[] state) {
        float[] s = new float[state.length];
        for (int i = 0; i < state.length; i++) {
            s[i] = (float)state[i];
        }
        float[] a = computeActions(s);
        double[] action = new double[a.length];
        for (int i = 0; i < action.length; i++) {
            action[i] = a[i];
        }
        return action;
    }

    /** Adapter from float to double array for {@link #computeDiscreteAction(float[])}. */
    default public long[] computeDiscreteAction(double[] state) {
        float[] s = new float[state.length];
        for (int i = 0; i < state.length; i++) {
            s[i] = (float)state[i];
        }
        return computeDiscreteAction(s);
    }

    /** Returns the continuous or discrete actions that should be performed in the given state. (Single Policy, Continuous or Discrete Actions) */
    float[] computeActions(float[] state);

    /** Returns the discrete actions that should be performed in the given state. (Single Policy, Tuple Decisions) */
    long[] computeDiscreteAction(float[] state);
}
