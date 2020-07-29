package ai.skymind.nativerl;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Method;
import java.net.URL;
import java.net.URLClassLoader;

/**
 *
 * @author saudet
 */
public interface ObservationFilter<O> {
    public static final String POLICY_CLASS_NAME = "PolicyObservationFilter";

    public static ObservationFilter load(File file) throws IOException, ReflectiveOperationException {
        return load(file, POLICY_CLASS_NAME);
    }
    public static ObservationFilter load(File file, String className) throws IOException, ReflectiveOperationException {
        if (!new File(file, className.replace('.', '/') + ".class").exists()) {
            return null;
        }
        ClassLoader classLoader = ObservationFilter.class.getClassLoader();
        try {
            // Java 8-
            Method method = classLoader.getClass().getDeclaredMethod("addURL", new Class[]{URL.class});
            method.setAccessible(true);
            method.invoke(classLoader, new Object[]{file.toURI().toURL()});
        } catch (NoSuchMethodException e) {
            // Java 9+, but requires java arguments "--add-opens java.base/jdk.internal.loader=ALL-UNNAMED"
            try {
                Method method = classLoader.getClass().getDeclaredMethod("appendToClassPathForInstrumentation", String.class);
                method.setAccessible(true);
                method.invoke(classLoader, file.getPath());
            } catch (RuntimeException e2) {
                throw new RuntimeException("Java arguments missing: \"--add-opens java.base/jdk.internal.loader=ALL-UNNAMED\"", e2);
            }
        }
        try {
            Class<? extends ObservationFilter> cls = Class.forName(className, true, classLoader).asSubclass(ObservationFilter.class);
            return cls.getDeclaredConstructor().newInstance();
        } catch (ClassNotFoundException e) {
            return null;
        }
    }

    double[] filter(O observations);
}
