package ai.skymind.nativerl;

import ai.skymind.nativerl.util.Reflect;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;

/**
 *
 * @author saudet
 */
public class RewardProcessor {
    public static final String METHOD_NAME = "rewardVariables";

    Class agentClass;
    Class rewardClass;
    Field[] rewardFields;
    Constructor rewardConstructor;
    boolean usesAgentId;

    public RewardProcessor(String agentClassName) throws ReflectiveOperationException {
        this(Class.forName(agentClassName, false, RewardProcessor.class.getClassLoader()));
    }
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

    public Class getRewardClass() {
        return rewardClass;
    }

    public String[] getVariableNames() {
        return Reflect.getFieldNames(rewardFields);
    }

    public double[] getVariables(Object agent) throws ReflectiveOperationException {
        return getVariables(agent, 0);
    }
    public double[] getVariables(Object agent, int agentId) throws ReflectiveOperationException {
        Object o = getRewardObject(agent, agentId);
        return toDoubles(getRewardObject(agent, agentId));
    }

    public <V> V getRewardObject(Object agent) throws ReflectiveOperationException {
        return getRewardObject(agent, 0);
    }
    public <V> V getRewardObject(Object agent, int agentId) throws ReflectiveOperationException {
        return usesAgentId ? (V)rewardConstructor.newInstance(agent, agentId) : (V)rewardConstructor.newInstance(agent);
    }

    public <V> double[] toDoubles(V rewardObject) throws ReflectiveOperationException {
        return Reflect.getFieldDoubles(rewardFields, rewardObject);
    }
}
