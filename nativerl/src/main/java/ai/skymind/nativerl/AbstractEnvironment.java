package ai.skymind.nativerl;

import java.lang.annotation.Annotation;
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
    protected Space actionSpace;
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
        actionSpace = getDiscreteSpace(discreteActions);
        actionMaskSpace = null;
        observationSpace = getContinuousSpace(continuousObservations);
        actionMask = new Array(new SSizeTVector().put(discreteActions));
        observation = new Array(new SSizeTVector().put(continuousObservations));
        reward = null;
        metrics = null;
    }

    /** Initializes all (protected) fields based on discreteActions and continuousObservations, but where annotations from agentClass can override them. */
    protected AbstractEnvironment(long discreteActions, long continuousObservations, Class agentClass) throws ReflectiveOperationException {
        this(discreteActions, continuousObservations);

        ActionProcessor ap = new ActionProcessor(agentClass);
        ActionMaskProcessor mp;
        try {
            mp = new ActionMaskProcessor(agentClass);
        } catch (ClassNotFoundException e) {
            mp = null;
        }
        Annotation[] actionSpaces = ap.getActionSpaces();
        if (actionSpaces.length == 1) {
            if (actionSpaces[0] instanceof ai.skymind.nativerl.annotation.Discrete) {
                ai.skymind.nativerl.annotation.Discrete d = (ai.skymind.nativerl.annotation.Discrete)actionSpaces[0];
                actionSpace = new Discrete(d.n(), d.size());
                actionMaskSpace = mp != null ? getContinuousSpace(0, 1, d.n() * d.size()) : null;
                actionMask = new Array(new SSizeTVector().put(d.n() * d.size()));
            } else if (actionSpaces[0] instanceof ai.skymind.nativerl.annotation.Continuous) {
                ai.skymind.nativerl.annotation.Continuous c = (ai.skymind.nativerl.annotation.Continuous)actionSpaces[0];
                FloatVector low, high;
                low = new FloatVector(c.low().length);
                for (int i = 0; i < c.low().length; i++) {
                    low.put(i, (float)c.low()[i]);
                }
                high = new FloatVector(c.high().length);
                for (int i = 0; i < c.high().length; i++) {
                    high.put(i, (float)c.high()[i]);
                }
                long length = (int)Arrays.stream(c.shape()).reduce((x, y) -> x * y).getAsLong();
                actionSpace = new Continuous(low, high, new SSizeTVector(c.shape()));
                actionMaskSpace = mp != null ? getContinuousSpace(0, 1, length) : null;
                actionMask = new Array(new SSizeTVector().put(length));
            }
        } else if (actionSpaces.length > 1) {
            throw new IllegalArgumentException("Only one action space is currently supported.");
        }
        // else assume 1 discrete action with n specified in constructor
    }

    /** Returns {@link #actionSpace}. */
    @Override public Space getActionSpace() {
        return actionSpace;
    }

    @Override public Space getActionMaskSpace() {
        return actionMaskSpace;
    }

    /** Returns {@link #observationSpace}. */
    @Override public Space getObservationSpace() {
        return observationSpace;
    }

    /** Returns {@link #actionMask}. */
    @Override public Array getActionMask() {
        return actionMask;
    }

    /** Returns {@link #observation}. */
    @Override public Array getObservation() {
        return observation;
    }

    @Override public Array getMetrics() {
        return metrics;
    }
}
