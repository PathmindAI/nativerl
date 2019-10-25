package ai.skymind.nativerl;

public abstract class AbstractEnvironment extends Environment {
    public static Discrete getDiscreteSpace(long n) {
        return new Discrete(n);
    }

    public static Continuous getContinuousSpace(long n) {
        return getContinuousSpace(-1.0f, 1.0f, n);
    }

    public static Continuous getContinuousSpace(float low, float high, long n) {
        return new Continuous(new FloatVector(low), new FloatVector(high), new SSizeTVector().put(n));
    }

    protected Space actionSpace;
    protected Space observationSpace;
    protected Array observation;
    protected Array reward;

    protected AbstractEnvironment(Space actionSpace, Space observationSpace) {
        this.actionSpace = actionSpace;
        this.observationSpace = observationSpace;
        observation = new Array(observationSpace.asContinuous().shape());
        reward = null;
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
}
