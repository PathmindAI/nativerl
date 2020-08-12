package ai.skymind.nativerl;

import ai.skymind.nativerl.util.Reflect;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;

/**
 * Finds the class containing values for the observations within the agent class,
 * and provides a few methods to obtain information about its fields as well as to access them.
 *
 * @author saudet
 */
public class ObservationProcessor {
    /** The name of the method where the local inner class needs to be defined in. */
    public static final String METHOD_NAME = "observations";

    Class agentClass;
    Class observationClass;
    Field[] observationFields;
    Constructor observationConstructor;
    boolean usesAgentId;

    /** Calls {@code this(Class.forName(agentClassName, false, this.getClassLoader()))}. */
    public ObservationProcessor(String agentClassName) throws ReflectiveOperationException {
        this(Class.forName(agentClassName, false, ObservationProcessor.class.getClassLoader()));
    }
    /** Looks inside the {@link #METHOD_NAME} method of the agent class given. */
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

    /** Returns the class we found within the {@link #METHOD_NAME} method of the agent class. */
    public Class getObservationClass() {
        return observationClass;
    }

    /** Returns {@code getObservationNames(agent, 0)}. */
    public String[] getObservationNames(Object agent) throws ReflectiveOperationException {
        return getObservationNames(agent, 0);
    }
    /** Returns {@code toNames(getObservationObject(agent, agentId))}. */
    public String[] getObservationNames(Object agent, int agentId) throws ReflectiveOperationException {
        return toNames(getObservationObject(agent, agentId));
    }

    /** Returns {@code getObservations(agent, 0)}. */
    public double[] getObservations(Object agent) throws ReflectiveOperationException {
        return getObservations(agent, 0);
    }
    /** Returns {@code toDoubles(getObservationObject(agent, agentId))}. */
    public double[] getObservations(Object agent, int agentId) throws ReflectiveOperationException {
        return toDoubles(getObservationObject(agent, agentId));
    }

    /** Returns {@code getObservationObject(agent, 0)}. */
    public <O> O getObservationObject(Object agent) throws ReflectiveOperationException {
        return getObservationObject(agent, 0);
    }
    /** Returns a new instance of the observation class, passing the given agentId to the constructor. */
    public <O> O getObservationObject(Object agent, int agentId) throws ReflectiveOperationException {
        return usesAgentId ? (O)observationConstructor.newInstance(agent, agentId) : (O)observationConstructor.newInstance(agent);
    }

    /** Returns the values that was assigned to the fields, with arrays flattened to doubles. */
    public <O> double[] toDoubles(O observationObject) throws ReflectiveOperationException {
        return Reflect.getFieldDoubles(observationFields, observationObject);
    }
    /** Returns the names of the fields in the order listed within the class found, with arrays flattened and suffixed with [0], [1], etc. */
    public <O> String[] toNames(O observationObject) throws ReflectiveOperationException {
        return Reflect.getFieldNames(observationFields, observationObject);
    }
}
