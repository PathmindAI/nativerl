package ai.skymind.nativerl;

import ai.skymind.nativerl.annotation.Discrete;
import ai.skymind.nativerl.annotation.Continuous;
import java.lang.annotation.Annotation;
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
        @Continuous(shape = {2}) float[] action3;
        @Continuous(shape = {2, 2}) double[] action4;

        public void doIt() {
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
        } catch (ReflectiveOperationException ex) {
            fail(ex.getMessage());
        }
    }
}
