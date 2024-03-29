package ai.skymind.nativerl.annotation;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Defines a tuple of {@link #size()} discrete action spaces with {@link #n()} actions each.
 *
 * @author saudet
 */
@Documented
@Retention(RetentionPolicy.RUNTIME)
@Target({ElementType.FIELD})
public @interface Discrete {
    long n();
    long size() default 1;
}
