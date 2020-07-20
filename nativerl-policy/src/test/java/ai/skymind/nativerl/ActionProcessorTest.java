package ai.skymind.nativerl;

import ai.skymind.nativerl.annotation.Discrete;
import ai.skymind.nativerl.annotation.Continuous;
import java.lang.annotation.Annotation;
import java.util.Random;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 *
 * @author saudet
 */
public class ActionProcessorTest {
    boolean didIt = false;

    class TestActions {
        @Discrete(n = 50) int action1;
        @Discrete(n = 50, size = 2) long[] action2;
        @Continuous(low = {10, 20}, high = {30, 40}, shape = 2) float[] action3;
        @Continuous(low = 0, high = 1, shape = {2, 2}) double[] action4;

        public void doIt() {
            if (didIt) {
                // random values
                assertTrue(action1 >= 0 && action1 < 50);
                assertTrue(action2[0] >= 0 && action2[0] < 50);
                assertTrue(action2[1] >= 0 && action2[1] < 50);
                assertTrue(action3[0] >= 10 && action3[0] < 30);
                assertTrue(action3[1] >= 20 && action3[1] < 40);
                assertTrue(action4[0] >= 0.0 && action4[0] < 1.0);
                assertTrue(action4[1] >= 0.0 && action4[1] < 1.0);
                assertTrue(action4[2] >= 0.0 && action4[2] < 1.0);
                assertTrue(action4[3] >= 0.0 && action4[3] < 1.0);
                return;
            }
            assertEquals(37, action1);
            assertArrayEquals(new long[] {42, 64}, action2);
            assertArrayEquals(new float[] {1, 2}, action3, 0);
            assertArrayEquals(new double[] {3, 4, 5, 6}, action4, 0);
            didIt = true;
        }
    }

    void actions() {
        class DummyActions extends TestActions {
        }
    }

    @Test public void testActions() {
        try {
            ActionProcessor ap = new ActionProcessor(this.getClass());
            assertEquals("DummyActions", ap.getActionClass().getSimpleName());
            assertArrayEquals(new String[] {"action1", "action2", "action3", "action4"}, ap.getActionNames());
            Annotation[] spaces = ap.getActionSpaces();
            assertEquals(4, spaces.length);
            assertEquals(50, ((Discrete)spaces[0]).n());
            assertEquals(50, ((Discrete)spaces[1]).n());
            assertArrayEquals(new long[] {2}, ((Continuous)spaces[2]).shape());
            assertArrayEquals(new long[] {2, 2}, ((Continuous)spaces[3]).shape());
            ap.doActions(this, new double[] {37, 42, 64, 1, 2, 3, 4, 5, 6});
            assertTrue(didIt);
            for (int i = 0; i < 100; i++) {
                ap.doActionsRandom(this, new Random(i));
            }
        } catch (ReflectiveOperationException ex) {
            fail(ex.getMessage());
        }
    }
}
