package ai.skymind.nativerl.annotation;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Defines a continuous space with given {@link #shape()} and with values between {@link #low()} and {@link #high()}.
 *
 * @author saudet
 */
@Documented
@Retention(RetentionPolicy.RUNTIME)
@Target({ElementType.FIELD})
public @interface Continuous {
    double[] low();
    double[] high();
    long[] shape();
}
