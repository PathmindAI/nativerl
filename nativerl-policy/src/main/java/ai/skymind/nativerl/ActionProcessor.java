package ai.skymind.nativerl;

import ai.skymind.nativerl.util.Reflect;
import java.lang.annotation.Annotation;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;

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

    public ActionProcessor(Class agentClass) throws ReflectiveOperationException {
        this.agentClass = agentClass;
        this.actionClass = Reflect.findLocalClass(agentClass, METHOD_NAME);
        this.actionFields = Reflect.getFields(actionClass);
        this.actionMethod = Reflect.getVoidMethod(actionClass);
        this.actionConstructor = actionClass.getDeclaredConstructor(agentClass);
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
        Object a = actionConstructor.newInstance(agent);
        Reflect.setFieldDoubles(actionFields, a, actions);
        actionMethod.invoke(a);
    }
}
