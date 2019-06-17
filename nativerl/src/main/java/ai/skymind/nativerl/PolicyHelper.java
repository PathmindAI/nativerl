package ai.skymind.nativerl;

public interface PolicyHelper {

    float[] computeContinuousAction(float[] state);

    long computeDiscreteAction(float[] state);
}
