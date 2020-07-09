package ai.skymind.nativerl;

import java.io.IOException;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 *
 * @author saudet
 */
public class ActionMaskProcessorTest {
    boolean data1 = true;
    boolean data2 = false;
    boolean data3[] = {false, true};

    class TestActionMasks {
        boolean mask1 = data1;
        boolean mask2 = data2;
        boolean[] mask3 = data3;
    }

    void getActionsMasks() {
        class DummyActionMasks extends TestActionMasks {
        }
    }

    @Test public void testActionMasks() {
        try {
            ActionMaskProcessor ap = new ActionMaskProcessor(this.getClass());
            assertEquals("DummyActionMasks", ap.getActionMaskClass().getSimpleName());
            assertArrayEquals(new String[] {"mask1", "mask2", "mask3"}, ap.getActionMaskNames());
            assertArrayEquals(new boolean[] {true, false, false, true}, ap.getActionMasks(this));
        } catch (ReflectiveOperationException | IOException ex) {
            fail(ex.getMessage());
        }
    }
}
