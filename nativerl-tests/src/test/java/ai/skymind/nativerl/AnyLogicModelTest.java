package ai.skymind.nativerl;

import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import org.bytedeco.javacpp.Loader;
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
public class AnyLogicModelTest extends ModelTest {

    @Test public void testTrafficPhases() throws Exception {
        File binDir = new File("target/dependency/nativerl-bin/");
        File exportDir = folder.newFolder("TrafficPhases");
        File modelDir = new File(getClass().getResource("trafficphases").toURI());
        File simulationDir = new File(exportDir, "TrafficPhases_Simulation");

        execute(modelDir, "anylogic", "-e", "-o", exportDir.getAbsolutePath(), modelDir.getAbsolutePath() + "/TrafficPhases.alp");

        copy(binDir, simulationDir);
        copy(new File(binDir, "examples/traintraffic.sh"), simulationDir);
        execute(simulationDir, "bash", "traintraffic.sh");

        File[] savedModels = find(simulationDir, "saved_model.pb");
        assertTrue(savedModels.length > 0);
        for (File f : savedModels) {
            File d = f.getParentFile();
            PolicyHelper h = PolicyHelper.load(d);
            double[] o = h.computeActions(new double[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
            System.out.println(Arrays.toString(o));
            assertEquals(o.length, 1);
            assertThat(o[0], anyOf(is(0.0), is(1.0)));

            Map<String, String> e = new HashMap<String, String>();
            e.put("NATIVERL_POLICY", d.getAbsolutePath());
            String p = Loader.getPlatform();
            if (p.startsWith("linux")) {
                execute(simulationDir, e, "bash", "TrafficPhases_linux.sh");
            } else if (p.startsWith("macosx")) {
                execute(simulationDir, e, "bash", "TrafficPhases_mac");
            } else if (p.startsWith("windows")) {
                execute(simulationDir, e, "cmd.exe", "/c", "TrafficPhases_windows.bat");
            } else {
                fail("Unsupported platform: " + p);
            }
        }
    }
}
