package ai.skymind.nativerl.util;

import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.annotation.PropertyAccessor;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * A simple object mapper holder for
 * using one single {@link ObjectMapper}
 * across the whole project.
 */
public class ObjectMapperHolder {
    private static ObjectMapper objectMapper = getMapper();

    private ObjectMapperHolder() {
    }

    /**
     * Get a single object mapper for use
     * with reading and writing json
     *
     * @return
     */
    public static ObjectMapper getJsonMapper() {
        return objectMapper;
    }

    private static ObjectMapper getMapper() {
        ObjectMapper om = new ObjectMapper();
        om.setVisibility(PropertyAccessor.FIELD, JsonAutoDetect.Visibility.ANY);

        return om;
    }
}
