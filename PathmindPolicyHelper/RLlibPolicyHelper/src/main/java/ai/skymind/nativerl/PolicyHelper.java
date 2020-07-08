package ai.skymind.nativerl;

public interface PolicyHelper {
    static final boolean disablePolicyHelper = Boolean.getBoolean("ai.skymind.nativerl.disablePolicyHelper");

    // Single Policy, Continuous Actions (Not yet implemented)
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

    // Single Policy, Tuple Decisions
    default public long[] computeDiscreteAction(double[] state) {
        float[] s = new float[state.length];
        for (int i = 0; i < state.length; i++) {
            s[i] = (float)state[i];
        }
        return computeDiscreteAction(s);
    }

    // Single Policy, Multi-Agent (No Longer Used)
    // default public int[] computeDiscreteAction(double[][] state) {
    //     int[] actions = new int[state.length];
    //     for (int i = 0; i < state.length; i++) {
    //         float[] s = new float[state[i].length];
    //         for (int j = 0; j < state[i].length; j++) {
    //             s[j] = (float)state[i][j];
    //         }
    //         actions[i] = (int)computeDiscreteAction(s);
    //     }
    //     return actions;
    // }

    float[] computeContinuousAction(float[] state);

    long[] computeDiscreteAction(float[] state);
}
