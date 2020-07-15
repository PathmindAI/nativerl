package ai.skymind.nativerl;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 *
 * @author saudet
 */
public class ObservationProcessorTest {
    static class TestFilter implements ObservationFilter<TestObservations> {
        public double[] filter(TestObservations observations) {
            return observations.obs3;
        }
    }

    int data1 = 37;
    float data2 = 42;
    double[] data3 = {1, 2, 3, 4, 5};

    class TestObservations {
        int obs1 = data1;
        float obs2 = data2;
        double[] obs3 = data3;
    }

    void observations() {
        class DummyObservations extends TestObservations {
        }
    }

    @Test public void testObservations() {
        try {
            ObservationProcessor op = new ObservationProcessor(this.getClass());
            assertEquals("DummyObservations", op.getObservationClass().getSimpleName());
            assertArrayEquals(new String[] {"obs1", "obs2", "obs3"}, op.getObservationNames());
            assertArrayEquals(new double[] {37, 42, 1, 2, 3, 4, 5}, op.getObservations(this, null), 0.0);
            assertArrayEquals(new double[] {1, 2, 3, 4, 5}, op.getObservations(this, new TestFilter()), 0.0);
        } catch (ReflectiveOperationException ex) {
            fail(ex.getMessage());
        }
    }
}
