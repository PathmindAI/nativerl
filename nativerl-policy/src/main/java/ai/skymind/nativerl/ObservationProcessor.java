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

    public double[] getObservations(Object agent, ObservationFilter filter) throws ReflectiveOperationException {
        return getObservations(agent, filter, 0);
    }
    public double[] getObservations(Object agent, ObservationFilter filter, int agentId) throws ReflectiveOperationException {
        Object o = usesAgentId ? observationConstructor.newInstance(agent, agentId) : observationConstructor.newInstance(agent);
        return filter != null ? filter.filter(o) : Reflect.getFieldDoubles(observationFields, o);
    }
}
