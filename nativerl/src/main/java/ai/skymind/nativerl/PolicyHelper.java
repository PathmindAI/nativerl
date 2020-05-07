package ai.skymind.nativerl;

/**
 * This is an interface that needs to be implemented by helper classes
 * that help users execute already trained reinforcement learning policies.
 * We can disable it at runtime by setting the "ai.skymind.nativerl.disablePolicyHelper"
 * system property to true, for example, during training.
 */
public interface PolicyHelper {
    static final boolean disablePolicyHelper = Boolean.getBoolean("ai.skymind.nativerl.disablePolicyHelper");

    /** Adapter from float to double array for {@link #computeContinuousAction(float[])}. */
    default public double[] computeContinuousAction(double[] state) {
        float[] s = new float[state.length];
        for (int i = 0; i < state.length; i++) {
            s[i] = (float)state[i];
        }
        float[] a = computeContinuousAction(s);
        double[] action = new double[a.length];
        for (int i = 0; i < action.length; i++) {
            action[i] = a[i];
        }
        return action;
    }

    /** Adapter from float to long for {@link #computeDiscreteAction(float[])}. */
    default public long computeDiscreteAction(double[] state) {
        float[] s = new float[state.length];
        for (int i = 0; i < state.length; i++) {
            s[i] = (float)state[i];
        }
        return computeDiscreteAction(s);
    }

    default public int[] computeDiscreteAction(double[][] state) {
        int[] actions = new int[state.length];
        for (int i = 0; i < state.length; i++) {
            float[] s = new float[state[i].length];
            for (int j = 0; j < state[i].length; j++) {
                s[j] = (float)state[i][j];
            }
            actions[i] = (int)computeDiscreteAction(s);
        }
        return actions;
    }

    /** Returns the continuous action that should be performed in the given state. */
    float[] computeContinuousAction(float[] state);

    /** Returns the discrete action that should be performed in the given state. */
    long computeDiscreteAction(float[] state);
}
