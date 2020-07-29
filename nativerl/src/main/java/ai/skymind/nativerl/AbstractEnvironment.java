package ai.skymind.nativerl;

import java.util.Arrays;

/**
 * Provides a few utility methods on top of the Environment interface.
 */
public abstract class AbstractEnvironment extends Environment {

     /** Returns {@code new Discrete(n)}. */
    public static Discrete getDiscreteSpace(long n) {
        return new Discrete(n);
    }

    /** Returns elements with values in range [-Inf, Inf]. */
    public static Continuous getContinuousSpace(long n) {
        return getContinuousSpace(Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY, n);
    }

    /** Returns {@code new Continuous(new FloatVector(low), new FloatVector(high), new SSizeTVector().put(n))},
     * that is a single dimensional vector space with n elements and all their values in range [low, high]. */
    public static Continuous getContinuousSpace(float low, float high, long n) {
        return new Continuous(new FloatVector(low), new FloatVector(high), new SSizeTVector().put(n));
    }

    /** The action Space. */
    protected Space[] actionSpaces;
    /** The action mask Space. */
    protected Space actionMaskSpace;
    /** The state Space. */
    protected Space observationSpace;
    /** The Array returned by the getActionMask() method. */
    protected Array actionMask;
    /** The Array returned by the getObservation() method. */
    protected Array observation;
    /** The Array returned by the step() method in the case of multiple agents. */
    protected Array reward;
    protected Array metrics;

    /** Initializes all (protected) fields based on discreteActions and continuousObservations. */
    protected AbstractEnvironment(long discreteActions, long continuousObservations) {
        actionSpaces = new Space[] {getDiscreteSpace(discreteActions)};
        actionMaskSpace = getContinuousSpace(0, 1, discreteActions);
        observationSpace = getContinuousSpace(continuousObservations);
        actionMask = new Array(new SSizeTVector().put(discreteActions));
        observation = new Array(new SSizeTVector().put(continuousObservations));
        reward = null;
        metrics = null;
    }

    /** Initializes all (protected) fields based on discreteActions and continuousObservations, but where annotations from agentClass can override them. */
    protected AbstractEnvironment(long discreteActions, long continuousObservations, Class agentClass) throws ReflectiveOperationException {
        this(discreteActions, continuousObservations);

        long actionMaskSize = 0;
        ActionProcessor ap = new ActionProcessor(agentClass);
        AnnotationProcessor[] as = ap.getActionSpaces();
        actionSpaces = new Space[as.length];
        for (int i = 0; i < actionSpaces.length; i++) {
            if (as[i].discrete) {
                AnnotationProcessor d = as[i];
                actionMaskSize += d.n * d.size;
                actionSpaces[i] = new Discrete(d.n, d.size);
            } else if (as[i].continuous) {
                AnnotationProcessor c = as[i];
                FloatVector low, high;
                low = new FloatVector(c.low.length);
                for (int j = 0; j < c.low.length; j++) {
                    low.put(j, (float)c.low[j]);
                }
                high = new FloatVector(c.high.length);
                for (int j = 0; j < c.high.length; j++) {
                    high.put(j, (float)c.high[j]);
                }
                actionMaskSize += (int)Arrays.stream(c.shape).reduce((x, y) -> x * y).getAsLong();
                actionSpaces[i] = new Continuous(low, high, new SSizeTVector(c.shape));
            }
        }

        try {
            ActionMaskProcessor mp = new ActionMaskProcessor(agentClass);
            actionMaskSpace = getContinuousSpace(0, 1, actionMaskSize);
            actionMask = new Array(new SSizeTVector().put(actionMaskSize));
        } catch (ClassNotFoundException e) {
            actionMaskSpace = null;
            actionMask = null;
        }
        // else assume 1 discrete action with n specified in constructor
    }

    /** Returns {@link #actionSpace}. */
    @Override public Space getActionSpace(long i) {
        return i < actionSpaces.length ? actionSpaces[(int)i] : null;
    }

    /** Returns {@link #actionMaskSpace}. */
    @Override public Space getActionMaskSpace() {
        return actionMaskSpace;
    }

    /** Returns {@link #observationSpace}. */
    @Override public Space getObservationSpace() {
        return observationSpace;
    }

    /** Returns 1. */
    @Override public long getNumberOfAgents() {
        return 1;
    }

    /** Returns {@code getActionMask(0)}. */
    public Array getActionMask() {
        return getActionMask(0);
    }

    /** Returns {@link #actionMask}. */
    @Override public Array getActionMask(long agentId) {
        return actionMask;
    }

    /** Returns {@code getObservation(0)}. */
    public Array getObservation() {
        return getObservation(0);
    }

    /** Returns {@link #observation}. */
    @Override public Array getObservation(long agentId) {
        return observation;
    }

    /** Returns {@code getMetrics(0)}. */
    public Array getMetrics() {
        return getMetrics(0);
    }

    @Override public Array getMetrics(long agentId) {
        return metrics;
    }

    /** Returns {@code isDone(-1)}, that is for all agents. */
    public boolean isDone() {
        return isDone(-1);
    }
}
