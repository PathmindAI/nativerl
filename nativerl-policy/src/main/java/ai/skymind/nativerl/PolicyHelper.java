package ai.skymind.nativerl;

public interface PolicyHelper {
    static final boolean disablePolicyHelper = Boolean.getBoolean("ai.skymind.nativerl.disablePolicyHelper");

    // Single Policy, Continuous or Discrete Actions
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

    // Single Policy, Tuple Decisions
    default public long[] computeDiscreteAction(double[] state) {
        float[] s = new float[state.length];
        for (int i = 0; i < state.length; i++) {
            s[i] = (float)state[i];
        }
        return computeDiscreteAction(s);
    }

    float[] computeActions(float[] state);

    long[] computeDiscreteAction(float[] state);
}
