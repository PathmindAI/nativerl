package ai.skymind.nativerl;

import ai.skymind.nativerl.util.Reflect;
import java.lang.annotation.Annotation;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.Random;

/**
 *
 * @author saudet
 */
public class ActionProcessor {
    public static final String METHOD_NAME = "actions";

    Class agentClass;
    Class actionClass;
    Field[] actionFields;
    Method actionMethod;
    Constructor actionConstructor;
    boolean usesAgentId;

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

    public Class getActionClass() {
        return actionClass;
    }

    public String[] getActionNames() {
        return Reflect.getFieldNames(actionFields);
    }

    public Annotation[] getActionSpaces() {
        return Reflect.getFieldAnnotations(actionFields);
    }

    public void doActions(Object agent, double[] actions) throws ReflectiveOperationException {
        doActions(agent, actions, 0);
    }
    public void doActions(Object agent, double[] actions, int agentId) throws ReflectiveOperationException {
        doActions(agent, actions, null, agentId);
    }

    public void doActionsRandom(Object agent, Random random) throws ReflectiveOperationException {
        doActionsRandom(agent, random, 0);
    }
    public void doActionsRandom(Object agent, Random random, int agentId) throws ReflectiveOperationException {
        doActions(agent, null, random, agentId);
    }

    public void doActions(Object agent, double[] actions, Random random) throws ReflectiveOperationException {
        doActions(agent, actions, random, 0);
    }
    public void doActions(Object agent, double[] actions, Random random, int agentId) throws ReflectiveOperationException {
        Object a = usesAgentId ? actionConstructor.newInstance(agent, agentId) : actionConstructor.newInstance(agent);
        Reflect.setFieldDoubles(actionFields, a, actions, random);
        actionMethod.invoke(a);
    }
}
