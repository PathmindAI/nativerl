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

    public RewardProcessor(Class agentClass) throws ReflectiveOperationException {
        this.agentClass = agentClass;
        this.rewardClass = Reflect.findLocalClass(agentClass, METHOD_NAME);
        this.rewardFields = Reflect.getFields(rewardClass);
        this.rewardConstructor = rewardClass.getDeclaredConstructor(agentClass);
        this.rewardConstructor.setAccessible(true);
    }

    public Class getRewardClass() {
        return rewardClass;
    }

    public String[] getVariableNames() {
        return Reflect.getFieldNames(rewardFields);
    }

    public double[] getVariables(Object agent) throws ReflectiveOperationException {
        Object o = rewardConstructor.newInstance(agent);
        return Reflect.getFieldDoubles(rewardFields, o);
    }

    public <V> V getRewardObject(Object agent) throws ReflectiveOperationException {
        return (V)rewardConstructor.newInstance(agent);
    }
}
