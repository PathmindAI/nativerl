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

    public ActionMaskProcessor(Class agentClass) throws ReflectiveOperationException {
        this.agentClass = agentClass;
        this.actionMaskClass = Reflect.findLocalClass(agentClass, METHOD_NAME);
        this.actionMaskFields = Reflect.getFields(actionMaskClass);
        this.actionMaskConstructor = actionMaskClass.getDeclaredConstructor(agentClass);
        this.actionMaskConstructor.setAccessible(true);
    }

    public Class getActionMaskClass() {
        return actionMaskClass;
    }

    public String[] getActionMaskNames() {
        return Reflect.getFieldNames(actionMaskFields);
    }

    public boolean[] getActionMasks(Object agent) throws ReflectiveOperationException {
        Object o = actionMaskConstructor.newInstance(agent);
        return Reflect.getFieldBooleans(actionMaskFields, o);
    }
}
