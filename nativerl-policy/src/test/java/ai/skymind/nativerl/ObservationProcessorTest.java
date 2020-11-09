package ai.skymind.nativerl;

import org.junit.Test;

import static org.junit.Assert.*;

/**
 *
 * @author saudet
 */
public class ObservationProcessorTest {
    static class TestFilter implements ObservationFilter<TestObservations> {
        public double[] filter(TestObservations observations) {
            return observations.obs4;
        }
    }

    int data1 = 37;
    int[] data2 = {11, 15};
    double data3 = 42;
    double[] data4 = {1, 2, 3, 4, 5};
    boolean data5 = true;
    boolean data6 = false;

    class TestObservations {
        int obs1 = data1;
        int[] obs2 = data2;
        double obs3 = data3;
        double[] obs4 = data4;
        boolean obs5 = data5;
        boolean obs6 = data6;
    }

    void observations(int agentId) {
        class DummyObservations extends TestObservations {
            float obs7 = agentId;
        }
    }

    @Test public void testObservations() {
        try {
            ObservationProcessor op = new ObservationProcessor(this.getClass());
            assertEquals("DummyObservations", op.getObservationClass().getSimpleName());
            assertArrayEquals(new String[] {"obs1", "obs2[0]", "obs2[1]", "obs3", "obs4[0]", "obs4[1]", "obs4[2]", "obs4[3]", "obs4[4]", "obs5", "obs6", "obs7"},
                    op.getObservationNames(this));
            assertArrayEquals(new String[] {"int", "int", "int", "double", "double", "double", "double", "double", "double", "boolean", "boolean", "float"},
                    op.getObservationTypes(this));
            assertEquals(op.getObservationNames(this).length, op.getObservationTypes(this).length);
            assertArrayEquals(new double[] {37, 11, 15, 42, 1, 2, 3, 4, 5, 1, 0, 64}, op.getObservations(this, 64), 0.0);
            TestObservations o = op.getObservationObject(this, 64);
            assertArrayEquals(new double[] {1, 2, 3, 4, 5}, new TestFilter().filter(o), 0.0);
        } catch (ReflectiveOperationException ex) {
            fail(ex.getMessage());
        }
    }
}
