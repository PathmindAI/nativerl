package ai.skymind.nativerl;

import ai.skymind.nativerl.util.Reflect;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;

/**
 *
 * @author saudet
 */
public class ObservationProcessor {
    public static final String METHOD_NAME = "observations";

    Class agentClass;
    Class observationClass;
    Field[] observationFields;
    Constructor observationConstructor;
    boolean usesAgentId;

    public ObservationProcessor(String agentClassName) throws ReflectiveOperationException {
        this(Class.forName(agentClassName, false, ObservationProcessor.class.getClassLoader()));
    }
    public ObservationProcessor(Class agentClass) throws ReflectiveOperationException {
        this.agentClass = agentClass;
        this.observationClass = Reflect.findLocalClass(agentClass, METHOD_NAME);
        this.observationFields = Reflect.getFields(observationClass);
        try {
            this.observationConstructor = observationClass.getDeclaredConstructor(agentClass, long.class);
            this.usesAgentId = true;
        } catch (NoSuchMethodException e) {
            try {
                this.observationConstructor = observationClass.getDeclaredConstructor(agentClass, int.class);
                this.usesAgentId = true;
            } catch (NoSuchMethodException e2) {
                this.observationConstructor = observationClass.getDeclaredConstructor(agentClass);
                this.usesAgentId = false;
            }
        }
        this.observationConstructor.setAccessible(true);
    }

    public Class getObservationClass() {
        return observationClass;
    }

    public String[] getObservationNames() {
        return Reflect.getFieldNames(observationFields);
    }

    public double[] getObservations(Object agent) throws ReflectiveOperationException {
        return getObservations(agent, 0);
    }
    public double[] getObservations(Object agent, int agentId) throws ReflectiveOperationException {
        return toDoubles(getObservationObject(agent, agentId));
    }

    public <O> O getObservationObject(Object agent) throws ReflectiveOperationException {
        return getObservationObject(agent, 0);
    }
    public <O> O getObservationObject(Object agent, int agentId) throws ReflectiveOperationException {
        return usesAgentId ? (O)observationConstructor.newInstance(agent, agentId) : (O)observationConstructor.newInstance(agent);
    }

    public <O> double[] toDoubles(O observationObject) throws ReflectiveOperationException {
        return Reflect.getFieldDoubles(observationFields, observationObject);
    }
}
