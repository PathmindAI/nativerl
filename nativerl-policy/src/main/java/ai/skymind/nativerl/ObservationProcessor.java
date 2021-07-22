package ai.skymind.nativerl;

import ai.skymind.nativerl.util.ObjectMapperHolder;
import ai.skymind.nativerl.util.Reflect;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.type.MapType;

import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.util.LinkedHashMap;
import java.util.Map;

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

    ObjectMapper objectMapper;

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
        this.objectMapper = ObjectMapperHolder.getJsonMapper();
    }

    /** Returns the class we found within the {@link #METHOD_NAME} method of the agent class. */
    public Class getObservationClass() {
        return observationClass;
    }

    /** Returns the fields of the class we found within the {@link #METHOD_NAME} method of the agent class. */
    public Field[] getObservationFields() {
        return observationFields;
    }

    /** Returns {@code getObservationNames(agent, 0)}. */
    public String[] getObservationNames(Object agent) throws ReflectiveOperationException {
        return getObservationNames(agent, 0);
    }
    /** Returns {@code toNames(getObservationObject(agent, agentId))}. */
    public String[] getObservationNames(Object agent, int agentId) throws ReflectiveOperationException {
        return toNames(getObservationObject(agent, agentId));
    }

    /** Returns {@code getObservationTypes(agent, 0)}. */
    public String[] getObservationTypes(Object agent) throws ReflectiveOperationException {
        return getObservationTypes(agent, 0);
    }
    /** Returns {@code toTypes(getObservationObject(agent, agentId))}. */
    public String[] getObservationTypes(Object agent, int agentId) throws ReflectiveOperationException {
        return toTypes(getObservationObject(agent, agentId));
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
    /** Returns the types of the fields in the order listed within the class found */
    public <O> String[] toTypes(O observationObject) throws ReflectiveOperationException {
        return Reflect.getFieldTypes(observationFields, observationObject);
    }

    /** Returns the json string of the given observation object
     * if actionMask array is not null, it will be converted to double array and added at the first of json*/
    public <O> String toJsonString(O observationObject, boolean[] actionMasks) throws JsonProcessingException {
        if (actionMasks != null) {
            double[] doubles = new double[actionMasks.length];
            for (int i = 0; i < actionMasks.length; i++) {
                doubles[i] = actionMasks[i] ? 1.0 : 0.0;
            }
            MapType mapType = objectMapper.getTypeFactory().constructMapType(LinkedHashMap.class, String.class, Object.class);

            Map<String, Object> map = new LinkedHashMap<>();
            map.put("actionMask", doubles);
            map.putAll(objectMapper.convertValue(observationObject, mapType));

            return objectMapper.writeValueAsString(map);
        } else {
            return objectMapper.writeValueAsString(observationObject);
        }
    }
}
