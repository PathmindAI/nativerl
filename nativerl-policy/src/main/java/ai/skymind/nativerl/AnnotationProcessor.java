package ai.skymind.nativerl;

import java.lang.annotation.Annotation;

/**
 *
 * @author saudet
 */
public class AnnotationProcessor {
    public final Annotation annotation;
    public final boolean discrete, continuous;
    public final long n, size;
    public final double[] low, high;
    public final long[] shape;

    public AnnotationProcessor(Annotation annotation) throws ReflectiveOperationException {
        Class<? extends Annotation> type = annotation.annotationType();
        String name = type.getSimpleName();
        this.annotation = annotation;
        this.discrete = name.equals("Discrete");
        this.continuous = name.equals("Continuous");

        if (discrete) {
            this.n = (Long)type.getMethod("n").invoke(annotation);
            this.size = (Long)type.getMethod("size").invoke(annotation);
            this.low = this.high = null;
            this.shape = null;
        } else if (continuous) {
            this.n = this.size = -1;
            this.low = (double[])type.getMethod("low").invoke(annotation);
            this.high = (double[])type.getMethod("high").invoke(annotation);
            this.shape = (long[])type.getMethod("shape").invoke(annotation);
        } else {
            this.n = this.size = -1;
            this.low = this.high = null;
            this.shape = null;
        }
    }
}
