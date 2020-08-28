package ai.skymind.nativerl;

import ai.skymind.nativerl.util.Reflect;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;

/**
 * Finds the class containing values for the reward variables within the agent class,
 * and provides a few methods to obtain information about its fields as well as to access them.
 *
 * @author saudet
 */
public class RewardProcessor {
    /** The name of the method where the local inner class needs to be defined in. */
    public static final String METHOD_NAME = "rewardVariables";

    Class agentClass;
    Class rewardClass;
    Field[] rewardFields;
    Constructor rewardConstructor;
    boolean usesAgentId;

    /** Calls {@code this(Class.forName(agentClassName, false, this.getClassLoader()))}. */
    public RewardProcessor(String agentClassName) throws ReflectiveOperationException {
        this(Class.forName(agentClassName, false, RewardProcessor.class.getClassLoader()));
    }
    /** Looks inside the {@link #METHOD_NAME} method of the agent class given. */
    public RewardProcessor(Class agentClass) throws ReflectiveOperationException {
        this.agentClass = agentClass;
        this.rewardClass = Reflect.findLocalClass(agentClass, METHOD_NAME);
        this.rewardFields = Reflect.getFields(rewardClass);
        try {
            this.rewardConstructor = rewardClass.getDeclaredConstructor(agentClass, long.class);
            this.usesAgentId = true;
        } catch (NoSuchMethodException e) {
            try {
                this.rewardConstructor = rewardClass.getDeclaredConstructor(agentClass, int.class);
                this.usesAgentId = true;
            } catch (NoSuchMethodException e2) {
                this.rewardConstructor = rewardClass.getDeclaredConstructor(agentClass);
                this.usesAgentId = false;
            }
        }
        this.rewardConstructor.setAccessible(true);
    }

    /** Returns the class we found within the {@link #METHOD_NAME} method of the agent class. */
    public Class getRewardClass() {
        return rewardClass;
    }

    /** Returns the fields of the class we found within the {@link #METHOD_NAME} method of the agent class. */
    public Field[] getRewardFields() {
        return rewardFields;
    }

    /** Returns {@code getVariableNames(agent, 0)}. */
    public String[] getVariableNames(Object agent) throws ReflectiveOperationException {
        return getVariableNames(agent, 0);
    }
    /** Returns {@code toNames(getVariableNames(agent, agentId))}. */
    public String[] getVariableNames(Object agent, int agentId) throws ReflectiveOperationException {
        return toNames(getRewardObject(agent, agentId));
    }

    /** Returns {@code getVariables(agent, 0)}. */
    public double[] getVariables(Object agent) throws ReflectiveOperationException {
        return getVariables(agent, 0);
    }
    /** Returns {@code toDoubles(getRewardObject(agent, agentId))}. */
    public double[] getVariables(Object agent, int agentId) throws ReflectiveOperationException {
        return toDoubles(getRewardObject(agent, agentId));
    }

    /** Returns {@code getRewardObject(agent, 0)}. */
    public <V> V getRewardObject(Object agent) throws ReflectiveOperationException {
        return getRewardObject(agent, 0);
    }
    /** Returns a new instance of the reward variables class, passing the given agentId to the constructor. */
    public <V> V getRewardObject(Object agent, int agentId) throws ReflectiveOperationException {
        return usesAgentId ? (V)rewardConstructor.newInstance(agent, agentId) : (V)rewardConstructor.newInstance(agent);
    }

    /** Returns the values that was assigned to the fields, with arrays flattened to doubles. */
    public <V> double[] toDoubles(V rewardObject) throws ReflectiveOperationException {
        return Reflect.getFieldDoubles(rewardFields, rewardObject);
    }
    /** Returns the names of the fields in the order listed within the class found, with arrays flattened and suffixed with [0], [1], etc. */
    public <V> String[] toNames(V rewardObject) throws ReflectiveOperationException {
        return Reflect.getFieldNames(rewardFields, rewardObject);
    }
}
