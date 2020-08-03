package ai.skymind.nativerl;

/**
 * An interface that users can implement to compute the reward somehow using before and after state values.
 *
 * @author saudet
 */
public interface RewardFunction<V> {
    double reward(V before, V after);
}
