package ai.skymind.nativerl;

import ai.skymind.nativerl.util.Reflect;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.Random;

/**
 * Finds the class containing values for the actions within the agent class,
 * and provides a few methods to obtain information about its fields as well as to access them.
 *
 * @author saudet
 */
public class ActionProcessor {
    /** The name of the method where the local inner class needs to be defined in. */
    public static final String METHOD_NAME = "actions";

    Class agentClass;
    Class actionClass;
    Field[] actionFields;
    Method actionMethod;
    Constructor actionConstructor;
    boolean usesAgentId;

    /** Calls {@code this(Class.forName(agentClassName, false, this.getClassLoader()))}. */
    public ActionProcessor(String agentClassName) throws ReflectiveOperationException {
        this(Class.forName(agentClassName, false, ActionProcessor.class.getClassLoader()));
    }
    /** Looks inside the {@link #METHOD_NAME} method of the agent class given. */
    public ActionProcessor(Class agentClass) throws ReflectiveOperationException {
        this.agentClass = agentClass;
        this.actionClass = Reflect.findLocalClass(agentClass, METHOD_NAME);
        this.actionFields = Reflect.getFields(actionClass);
        this.actionMethod = Reflect.getVoidMethod(actionClass);
        try {
            this.actionConstructor = actionClass.getDeclaredConstructor(agentClass, long.class);
            this.usesAgentId = true;
        } catch (NoSuchMethodException e) {
            try {
                this.actionConstructor = actionClass.getDeclaredConstructor(agentClass, int.class);
                this.usesAgentId = true;
            } catch (NoSuchMethodException e2) {
                this.actionConstructor = actionClass.getDeclaredConstructor(agentClass);
                this.usesAgentId = false;
            }
        }
        this.actionConstructor.setAccessible(true);
    }

    /** Returns the class we found within the {@link #METHOD_NAME} method of the agent class. */
    public Class getActionClass() {
        return actionClass;
    }

    /** Returns {@code getActionNames(agent, 0)}. */
    public String[] getActionNames(Object agent) throws ReflectiveOperationException {
        return getActionNames(agent, 0);
    }
    /** Returns the names of the fields in the order listed within the class found, with arrays flattened and suffixed with [0], [1], etc. */
    public String[] getActionNames(Object agent, int agentId) throws ReflectiveOperationException {
        Object a = usesAgentId ? actionConstructor.newInstance(agent, agentId) : actionConstructor.newInstance(agent);
        return Reflect.getFieldNames(actionFields, a);
    }

    /** Returns the annotations (Discrete or Continuous) found on the fields in the order listed within the class found. */
    public AnnotationProcessor[] getActionSpaces() throws ReflectiveOperationException {
        return Reflect.getFieldAnnotations(actionFields);
    }

    /** Returns {@code getActions(agent, random, 0)}. */
    public double[] getActions(Object agent, Random random) throws ReflectiveOperationException {
        return getActions(agent, random, 0);
    }
    /** Assigns random values to the fields and returns with arrays flattened to doubles. */
    public double[] getActions(Object agent, Random random, int agentId) throws ReflectiveOperationException {
        Object a = usesAgentId ? actionConstructor.newInstance(agent, agentId) : actionConstructor.newInstance(agent);
        Reflect.setFieldDoubles(actionFields, a, null, random);
        return Reflect.getFieldDoubles(actionFields, a);
    }

    /** Calls {@code doActions(agent, actions, 0)}. */
    public void doActions(Object agent, double[] actions) throws ReflectiveOperationException {
        doActions(agent, actions, 0);
    }
    /** Assigns the values of the fields in the order listed in the class and calls the method with no parameters found in that action class. */
    public void doActions(Object agent, double[] actions, int agentId) throws ReflectiveOperationException {
        Object a = usesAgentId ? actionConstructor.newInstance(agent, agentId) : actionConstructor.newInstance(agent);
        Reflect.setFieldDoubles(actionFields, a, actions, null);
        actionMethod.invoke(a);
    }
}
