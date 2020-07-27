package ai.skymind.nativerl;

import ai.skymind.nativerl.util.Reflect;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;

/**
 *
 * @author saudet
 */
public class ActionMaskProcessor {
    public static final String METHOD_NAME = "actionMasks";

    Class agentClass;
    Class actionMaskClass;
    Field[] actionMaskFields;
    Constructor actionMaskConstructor;
    boolean usesAgentId;

    public ActionMaskProcessor(Class agentClass) throws ReflectiveOperationException {
        this.agentClass = agentClass;
        this.actionMaskClass = Reflect.findLocalClass(agentClass, METHOD_NAME);
        this.actionMaskFields = Reflect.getFields(actionMaskClass);
        try {
            this.actionMaskConstructor = actionMaskClass.getDeclaredConstructor(agentClass, long.class);
            this.usesAgentId = true;
        } catch (NoSuchMethodException e) {
            try {
                this.actionMaskConstructor = actionMaskClass.getDeclaredConstructor(agentClass, int.class);
                this.usesAgentId = true;
            } catch (NoSuchMethodException e2) {
                this.actionMaskConstructor = actionMaskClass.getDeclaredConstructor(agentClass);
                this.usesAgentId = false;
            }
        }
        this.actionMaskConstructor.setAccessible(true);
    }

    public Class getActionMaskClass() {
        return actionMaskClass;
    }

    public String[] getActionMaskNames() {
        return Reflect.getFieldNames(actionMaskFields);
    }

    public boolean[] getActionMasks(Object agent) throws ReflectiveOperationException {
        return getActionMasks(agent, 0);
    }
    public boolean[] getActionMasks(Object agent, int agentId) throws ReflectiveOperationException {
        Object o = usesAgentId ? actionMaskConstructor.newInstance(agent, agentId) : actionMaskConstructor.newInstance(agent);
        return Reflect.getFieldBooleans(actionMaskFields, o);
    }
}
