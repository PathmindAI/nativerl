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

    /** Adapter from float to double array for {@link #computeDiscreteAction(float[])}. */
    default public long computeDiscreteAction(double[] state) {
        float[] s = new float[state.length];
        for (int i = 0; i < state.length; i++) {
            s[i] = (float)state[i];
        }
        return computeDiscreteAction(s);
    }

    /** Calls {@link #computeContinuousAction(float[])} in a loop for all states and returns the actions. */
    default public double[][] computeContinuousAction(double[][] states) {
        double[][] actions = new double[states.length][];
        for (int i = 0; i < states.length; i++) {
            float[] s = new float[states[i].length];
            for (int j = 0; j < states[i].length; j++) {
                s[j] = (float)states[i][j];
            }
            float[] a = computeContinuousAction(s);
            actions[i] = new double[a.length];
            for (int j = 0; j < actions[i].length; j++) {
                actions[i][j] = a[j];
            }
        }
        return actions;
    }

    /** Calls {@link #computeDiscreteAction(float[])} in a loop for all states and returns the actions. */
    default public int[] computeDiscreteAction(double[][] states) {
        int[] actions = new int[states.length];
        for (int i = 0; i < states.length; i++) {
            float[] s = new float[states[i].length];
            for (int j = 0; j < states[i].length; j++) {
                s[j] = (float)states[i][j];
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
