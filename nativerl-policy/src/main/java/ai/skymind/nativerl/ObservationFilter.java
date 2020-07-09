package ai.skymind.nativerl;

/**
 *
 * @author saudet
 */
public interface ObservationFilter<O> {
    double[] filter(O observations);
}
