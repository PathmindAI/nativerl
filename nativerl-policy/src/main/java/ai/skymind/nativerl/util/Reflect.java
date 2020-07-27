package ai.skymind.nativerl.util;

import ai.skymind.nativerl.annotation.Continuous;
import ai.skymind.nativerl.annotation.Discrete;
import java.io.File;
import java.io.IOException;
import java.lang.annotation.Annotation;
import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Random;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;

/**
 *
 * @author saudet
 */
public class Reflect {

    public static Class findLocalClass(Class parentClass, String methodName) throws ClassNotFoundException {
        ArrayList<String> names = new ArrayList<String>();
        String packagePath = parentClass.getPackage().getName().replace('.', '/');
        File file;
        try {
            file = new File(parentClass.getProtectionDomain().getCodeSource().getLocation().toURI());
        } catch (IllegalArgumentException | URISyntaxException ex) {
            file = new File(parentClass.getProtectionDomain().getCodeSource().getLocation().getPath());
        }
        if (file.isDirectory()) {
            File[] files  = new File(file, packagePath).listFiles();
            for (File f : files) {
                names.add(packagePath + '/' + f.getName());
            }
        } else {
            try {
                try (JarFile jarFile = new JarFile(file)) {
                    Enumeration<JarEntry> entries = jarFile.entries();
                    while (entries.hasMoreElements()) {
                        JarEntry entry = entries.nextElement();
                        names.add(entry.getName());
                    }
                }
            } catch (IOException ex) {
                throw new ClassNotFoundException(ex.toString(), ex);
            }
        }
        for (String name : names) {
            if (name.endsWith(".class")) {
                String className = name.replace('/', '.').substring(0, name.length() - 6);
                Class c = Class.forName(className, false, parentClass.getClassLoader());
                Method m = c.getEnclosingMethod();
                if (m != null && m.getName().contains(methodName)) {
                    return c;
                }
            }
        }
        throw new ClassNotFoundException("Could not find local class in " + parentClass + "." + methodName);
    }

    public static Field[] getFields(Class cls) {
        ArrayList<Field> fields = new ArrayList<Field>();
        ArrayList<Class> classes = new ArrayList<Class>();
        while (cls != null && cls != Object.class) {
            classes.add(0, cls);
            cls = cls.getSuperclass();
        }
        for (Class c : classes) {
            for (Field f : c.getDeclaredFields()) {
                Class t = f.getType();
                if ((!t.isPrimitive() && !t.isArray()) || f.getName().startsWith("val$")) {
                    continue;
                }
                f.setAccessible(true);
                fields.add(f);
            }
        }
        return fields.toArray(new Field[fields.size()]);
    }

    public static Method getVoidMethod(Class cls) {
        ArrayList<Method> methods = new ArrayList<Method>();
        ArrayList<Class> classes = new ArrayList<Class>();
        while (cls != null && cls != Object.class) {
            classes.add(0, cls);
            cls = cls.getSuperclass();
        }
        for (Class c : classes) {
            for (Method m : c.getDeclaredMethods()) {
                if (m.getParameterCount() == 0 && m.getReturnType() == void.class) {
                    methods.add(m);
                }
            }
        }
        if (methods.size() == 0) {
            throw new IllegalArgumentException("No void parameter-less method found in " + cls);
        } else if (methods.size() > 1) {
            throw new IllegalArgumentException("More than one void parameter-less method found " + methods);
        }
        Method m = methods.get(0);
        m.setAccessible(true);
        return m;
    }

    public static Annotation getFieldAnnotation(Field field) {
        Annotation discrete = field.getAnnotation(Discrete.class);
        Annotation continuous = field.getAnnotation(Continuous.class);
        if (discrete != null && continuous == null) {
            return discrete;
        } else if (discrete == null && continuous != null) {
            return continuous;
        } else if (discrete != null && continuous != null) {
            throw new IllegalArgumentException("Field " + field + " cannot be both Discrete and Continuous");
        } else {
            throw new IllegalArgumentException("Field " + field + " must be annotated with Discrete or Continuous.");
        }
    }

    public static Annotation[] getFieldAnnotations(Field[] fields) {
        ArrayList<Annotation> annotations = new ArrayList<Annotation>();
        for (Field f : fields) {
            annotations.add(getFieldAnnotation(f));
        }
        return annotations.toArray(new Annotation[annotations.size()]);
    }

    public static String[] getFieldNames(Field[] fields) {
        ArrayList<String> names = new ArrayList<String>();
        for (Field f : fields) {
            names.add(f.getName());
        }
        return names.toArray(new String[names.size()]);
    }

    public static Object[] getFieldObjects(Field[] fields, Object object) throws ReflectiveOperationException {
        ArrayList<Object> objects = new ArrayList<Object>();
        for (Field f : fields) {
            Class t = f.getType();
            if (t.isArray()) {
                Object array = f.get(object);
                int length = Array.getLength(array);
                for (int i = 0; i < length; i++) {
                    objects.add(Array.get(array, i));
                }
            } else {
                objects.add(f.get(object));
            }
        }
        return objects.toArray(new Object[objects.size()]);
    }

    public static double[] getFieldDoubles(Field[] fields, Object object) throws ReflectiveOperationException {
        Object[] objects = getFieldObjects(fields, object);
        double[] doubles = new double[objects.length];
        for (int i = 0; i < doubles.length; i++) {
            doubles[i] = ((Number)objects[i]).doubleValue();
        }
        return doubles;
    }

    public static boolean[] getFieldBooleans(Field[] fields, Object object) throws ReflectiveOperationException {
        Object[] objects = getFieldObjects(fields, object);
        boolean[] booleans = new boolean[objects.length];
        for (int i = 0; i < booleans.length; i++) {
            booleans[i] = (Boolean)objects[i];
        }
        return booleans;
    }

    public static void setFieldDoubles(Field[] fields, Object object, double[] values, Random random) throws ReflectiveOperationException {
        int i = 0;
        for (Field f : fields) {
            Annotation a = getFieldAnnotation(f);
            int n = 0, length;
            double[] low = null, high = null;
            if (a instanceof Discrete) {
                n = (int)((Discrete)a).n();
                length = (int)((Discrete)a).size();
            } else {
                low = (double[])((Continuous)a).low();
                high = (double[])((Continuous)a).high();
                length = (int)Arrays.stream(((Continuous)a).shape()).reduce((x, y) -> x * y).getAsLong();
                if (low.length == 1 && length > 1) {
                    low = Arrays.copyOf(low, length);
                    Arrays.fill(low, low[0]);
                }
                if (high.length == 1 && length > 1) {
                    high = Arrays.copyOf(high, length);
                    Arrays.fill(high, high[0]);
                }
            }
            Class t = f.getType();
            if (t.isArray()) {
                Class c = t.getComponentType();
                Object array = Array.newInstance(c, length);
                f.set(object, array);
                if (c == int.class) {
                    for (int j = 0; j < length; j++) {
                        Array.setInt(array, j, values != null ? (int)values[i++] : random.nextInt(n));
                    }
                } else if (c == long.class) {
                    for (int j = 0; j < length; j++) {
                        Array.setLong(array, j, values != null ? (long)values[i++] : (long)(n * random.nextDouble()));
                    }
                } else if (c == float.class) {
                    for (int j = 0; j < length; j++) {
                        Array.setFloat(array, j, values != null ? (float)values[i++] : (float)((high[j] - low[j]) * random.nextDouble() + low[j]));
                    }
                } else if (c == double.class) {
                    for (int j = 0; j < length; j++) {
                        Array.setDouble(array, j, values != null ? values[i++] : (high[j] - low[j]) * random.nextDouble() + low[j]);
                    }
                } else {
                    throw new IllegalArgumentException("Field " + f + " must be int, long, float, or double.");
                }
            } else {
                if (t == int.class) {
                    f.setInt(object, values != null ? (int)values[i++] : random.nextInt(n));
                } else if (t == long.class) {
                    f.setLong(object, values != null ?  (long)values[i++] : (long)(n * random.nextDouble()));
                } else if (t == float.class) {
                    f.setFloat(object, values != null ? (float)values[i++] : (float)((high[0] - low[0]) * random.nextDouble() + low[0]));
                } else if (t == double.class) {
                    f.setDouble(object, values != null ? values[i++] : (high[0] - low[0]) * random.nextDouble() + low[0]);
                } else {
                    throw new IllegalArgumentException("Field " + f + " must be int, long, float, or double.");
                }
            }
        }
    }

}
