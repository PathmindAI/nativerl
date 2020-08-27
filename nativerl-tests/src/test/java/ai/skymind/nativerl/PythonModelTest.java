package ai.skymind.nativerl;

import java.io.File;
import java.util.Arrays;
import org.junit.Test;

import static org.hamcrest.core.AnyOf.anyOf;
import static org.hamcrest.core.Is.is;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 *
 * @author saudet
 */
public class PythonModelTest extends ModelTest {

    @Test public void testCartpole() throws Exception {
        File binDir = new File("target/dependency/nativerl-bin/");
        File modelDir = folder.newFolder("Cartpole");

        copy(binDir, modelDir);
        copy(new File(binDir, "examples/traincartpole.sh"), modelDir);
        execute(modelDir, "bash", "traincartpole.sh");

        File[] savedModels = find(modelDir, "saved_model.pb");
        assertTrue(savedModels.length > 0);
        for (File f : savedModels) {
            PolicyHelper h = PolicyHelper.load(f.getParentFile());
            double[] o = h.computeActions(new double[] {0, 1, 2, 3});
            System.out.println(Arrays.toString(o));
            assertEquals(o.length, 1);
            assertThat(o[0], anyOf(is(0.0), is(1.0)));
        }
    }
}
