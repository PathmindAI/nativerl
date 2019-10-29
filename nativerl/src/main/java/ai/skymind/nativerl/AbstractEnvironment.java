package ai.skymind.nativerl;

/**
 * Provides a few utility methods on top of the Environment interface.
 */
public abstract class AbstractEnvironment extends Environment {
    /** Returns {@code new Discrete(n)}. */
    public static Discrete getDiscreteSpace(long n) {
        return new Discrete(n);
    }

    /** Returns {@code getContinuousSpace(-1.0f, 1.0f, n)},
     * that is n elements with values in range [-1.0, 1.0]. */
    public static Continuous getContinuousSpace(long n) {
        return getContinuousSpace(-1.0f, 1.0f, n);
    }

    /* Returns {@code new Continuous(new FloatVector(low), new FloatVector(high), new SSizeTVector().put(n))},
     * that is a single dimensional vector space with n elements and all their values in range [low, high]. */
    public static Continuous getContinuousSpace(float low, float high, long n) {
        return new Continuous(new FloatVector(low), new FloatVector(high), new SSizeTVector().put(n));
    }

    /** The action Space. */
    protected Space actionSpace;
    /** The state Space. */
    protected Space observationSpace;
    /** The Array returned by the getObservation() method. */
    protected Array observation;
    /** The Array returned by the step() method in the case of multiple agents. */
    protected Array reward;

    /** Initializes all (protected) fields based on actionSpace and observationSpace. */
    protected AbstractEnvironment(Space actionSpace, Space observationSpace) {
        this.actionSpace = actionSpace;
        this.observationSpace = observationSpace;
        observation = new Array(observationSpace.asContinuous().shape());
        reward = null;
    }

    /** Returns {@link #actionSpace}. */
    @Override public Space getActionSpace() {
        return actionSpace;
    }

    /** Returns {@link #observationSpace}. */
    @Override public Space getObservationSpace() {
        return observationSpace;
    }

    /** Returns {@link #observation}. */
    @Override public Array getObservation() {
        return observation;
    }
}
