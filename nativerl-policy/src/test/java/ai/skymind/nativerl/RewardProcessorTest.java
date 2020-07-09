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
public class RewardProcessorTest {
    static class TestFunction implements RewardFunction<TestVariables> {
        public double reward(TestVariables before, TestVariables after) {
            return before.var1 + before.var2 + after.var3 + after.var4[4];
        }
    }

    int data1 = 37;
    long data2 = 42;
    float data3 = 64;
    double[] data4 = {1, 2, 3, 4, 5};

    class TestVariables {
        int var1 = data1;
        long var2 = data2;
        float var3 = data3;
        double[] var4 = data4;
    }

    void getRewardVariables() {
        class DummyVariables extends TestVariables {
        }
    }

    @Test public void testObservations() {
        try {
            RewardProcessor rp = new RewardProcessor(this.getClass());
            assertEquals("DummyVariables", rp.getRewardClass().getSimpleName());
            assertArrayEquals(new String[] {"var1", "var2", "var3", "var4"}, rp.getVariableNames());
            assertArrayEquals(new double[] {37, 42, 64, 1, 2, 3, 4, 5}, rp.getVariables(this), 0.0);
            TestVariables v = rp.getRewardObject(this);
            assertEquals(37 + 42 + 64 + 5, new TestFunction().reward(v, v), 0.0);
        } catch (ReflectiveOperationException | IOException ex) {
            fail(ex.getMessage());
        }
    }
}
