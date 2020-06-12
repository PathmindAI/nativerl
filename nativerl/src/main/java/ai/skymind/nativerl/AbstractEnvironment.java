package ai.skymind.nativerl;

public abstract class AbstractEnvironment extends Environment {
    public static Discrete getDiscreteSpace(long n) {
        return new Discrete(n);
    }

    public static Continuous getContinuousSpace(long n) {
        return getContinuousSpace(Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY, n);
    }

    public static Continuous getContinuousSpace(float low, float high, long n) {
        return new Continuous(new FloatVector(low), new FloatVector(high), new SSizeTVector().put(n));
    }

    protected Space actionSpace;
    protected Space observationSpace;
    protected Array observation;
    protected Array reward;
    protected Array metrics;

    protected AbstractEnvironment(long discreteActions, long continuousObservations) {
        actionSpace = getDiscreteSpace(discreteActions);
        observationSpace = getContinuousSpace(continuousObservations);
        observation = new Array(new SSizeTVector().put(continuousObservations));
        reward = null;
        metrics = null;
    }

    @Override public Space getActionSpace() {
        return actionSpace;
    }

    @Override public Space getObservationSpace() {
        return observationSpace;
    }

    @Override public Array getObservation() {
        return observation;
    }

    @Override public Array getMetrics() {
        return metrics;
    }
}
