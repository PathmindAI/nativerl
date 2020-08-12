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

    void observations(int agentId) {
        class DummyObservations extends TestObservations {
            double obs4 = agentId;
        }
    }

    @Test public void testObservations() {
        try {
            ObservationProcessor op = new ObservationProcessor(this.getClass());
            assertEquals("DummyObservations", op.getObservationClass().getSimpleName());
            assertArrayEquals(new String[] {"obs1", "obs2", "obs3[0]", "obs3[1]", "obs3[2]", "obs3[3]", "obs3[4]", "obs4"}, op.getObservationNames(this));
            assertArrayEquals(new double[] {37, 42, 1, 2, 3, 4, 5, 64}, op.getObservations(this, 64), 0.0);
            TestObservations o = op.getObservationObject(this, 64);
            assertArrayEquals(new double[] {1, 2, 3, 4, 5}, new TestFilter().filter(o), 0.0);
        } catch (ReflectiveOperationException ex) {
            fail(ex.getMessage());
        }
    }
}
