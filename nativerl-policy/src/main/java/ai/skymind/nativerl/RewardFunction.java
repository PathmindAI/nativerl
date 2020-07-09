package ai.skymind.nativerl;

/**
 *
 * @author saudet
 */
public interface RewardFunction<V> {
    double reward(V before, V after);
}
