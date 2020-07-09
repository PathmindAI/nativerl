package ai.skymind.nativerl;

import ai.skymind.nativerl.util.Reflect;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;

/**
 *
 * @author saudet
 */
public class ObservationProcessor {
    public static final String METHOD_NAME = "getObservations";

    Class agentClass;
    Class observationClass;
    Field[] observationFields;
    Constructor observationConstructor;

    public ObservationProcessor(Class agentClass) throws ReflectiveOperationException, IOException {
        this.agentClass = agentClass;
        this.observationClass = Reflect.findLocalClass(agentClass, METHOD_NAME);
        this.observationFields = Reflect.getFields(observationClass);
        this.observationConstructor = observationClass.getDeclaredConstructor(agentClass);
    }

    public Class getObservationClass() {
        return observationClass;
    }

    public String[] getObservationNames() {
        return Reflect.getFieldNames(observationFields);
    }

    public double[] getObservations(Object agent, ObservationFilter filter) throws ReflectiveOperationException {
        Object o = observationConstructor.newInstance(agent);
        return filter != null ? filter.filter(o) : Reflect.getFieldDoubles(observationFields, o);
    }
}
