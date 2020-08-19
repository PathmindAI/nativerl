package ai.skymind.nativerl;

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

    void rewardVariables(int agentId) {
        class DummyVariables extends TestVariables {
            float var5 = agentId;
        }
    }

    @Test public void testObservations() {
        try {
            RewardProcessor rp = new RewardProcessor(this.getClass());
            assertEquals("DummyVariables", rp.getRewardClass().getSimpleName());
            assertArrayEquals(new String[] {"var1", "var2", "var3", "var4[0]", "var4[1]", "var4[2]", "var4[3]", "var4[4]", "var5"}, rp.getVariableNames(this));
            assertArrayEquals(new double[] {37, 42, 64, 1, 2, 3, 4, 5, 24}, rp.getVariables(this, 24), 0.0);
            TestVariables v = rp.getRewardObject(this, 24);
            assertEquals(37 + 42 + 64 + 5, new TestFunction().reward(v, v), 0.0);
        } catch (ReflectiveOperationException  ex) {
            fail(ex.getMessage());
        }
    }
}
