package ai.skymind.nativerl;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.net.URL;
import java.net.URLClassLoader;

/**
 * An interface that users can implement to filter observations somehow.
 * The loading mechanism currently needs java arguments "--add-opens java.base/jdk.internal.loader=ALL-UNNAMED" for Java 9+.
 *
 * @author saudet
 */
public interface ObservationFilter<O> {
    public static final String POLICY_CLASS_NAME = "PolicyObservationFilter";

    /** Returns {@code load(directory, POLICY_CLASS_NAME)}. */
    public static ObservationFilter load(File directory) throws IOException, ReflectiveOperationException {
        return load(directory, POLICY_CLASS_NAME);
    }
    /** Returns an instance of an implementation of ObservationFilter found in the given directory with the given class name. */
    public static ObservationFilter load(File directory, String className) throws IOException, ReflectiveOperationException {
        if (!new File(directory, className.replace('.', '/') + ".class").exists()) {
            return null;
        }
        ClassLoader classLoader = ObservationFilter.class.getClassLoader();
        try {
            // Java 8-
            Method method = classLoader.getClass().getDeclaredMethod("addURL", new Class[]{URL.class});
            method.setAccessible(true);
            method.invoke(classLoader, new Object[]{directory.toURI().toURL()});
        } catch (NoSuchMethodException e) {
            // Java 9+, but requires java arguments "--add-opens java.base/jdk.internal.loader=ALL-UNNAMED"
            try {
                Method method = classLoader.getClass().getDeclaredMethod("appendToClassPathForInstrumentation", String.class);
                method.setAccessible(true);
                method.invoke(classLoader, directory.getPath());
            } catch (RuntimeException e2) {
                throw new RuntimeException("Java arguments missing: \"--add-opens java.base/jdk.internal.loader=ALL-UNNAMED\"", e2);
            }
        }
        try {
            Class<? extends ObservationFilter> cls = Class.forName(className, true, classLoader).asSubclass(ObservationFilter.class);
            Constructor<? extends ObservationFilter> c = cls.getDeclaredConstructor();
            c.setAccessible(true);
            return c.newInstance();
        } catch (ClassNotFoundException e) {
            return null;
        }
    }

    double[] filter(O observations);
}
