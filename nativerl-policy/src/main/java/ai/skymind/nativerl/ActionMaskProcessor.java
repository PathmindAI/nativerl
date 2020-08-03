package ai.skymind.nativerl;

import ai.skymind.nativerl.util.Reflect;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;

/**
 * Finds the class containing values for the action masks within the agent class,
 * and provides a few methods to obtain information about its fields as well as to access them.
 *
 * @author saudet
 */
public class ActionMaskProcessor {
    /** The name of the method where the local inner class needs to be defined in. */
    public static final String METHOD_NAME = "actionMasks";

    Class agentClass;
    Class actionMaskClass;
    Field[] actionMaskFields;
    Constructor actionMaskConstructor;
    boolean usesAgentId;

    /** Calls {@code this(Class.forName(agentClassName, false, this.getClassLoader()))}. */
    public ActionMaskProcessor(String agentClassName) throws ReflectiveOperationException {
        this(Class.forName(agentClassName, false, ActionMaskProcessor.class.getClassLoader()));
    }
    /** Looks inside the {@link #METHOD_NAME} method of the agent class given. */
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

    /** Returns the class we found within the {@link #METHOD_NAME} method of the agent class. */
    public Class getActionMaskClass() {
        return actionMaskClass;
    }

    /** Returns the names of the fields in the order listed within the class found. */
    public String[] getActionMaskNames() {
        return Reflect.getFieldNames(actionMaskFields);
    }

    /** Returns {@code getActionMasks(agent, 0)}. */
    public boolean[] getActionMasks(Object agent) throws ReflectiveOperationException {
        return getActionMasks(agent, 0);
    }
    /** Returns the values of the fields in the order listed within the class found for the given agentId, with arrays flattened to booleans. */
    public boolean[] getActionMasks(Object agent, int agentId) throws ReflectiveOperationException {
        Object o = usesAgentId ? actionMaskConstructor.newInstance(agent, agentId) : actionMaskConstructor.newInstance(agent);
        return Reflect.getFieldBooleans(actionMaskFields, o);
    }
}
