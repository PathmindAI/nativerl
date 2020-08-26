package ai.skymind.nativerl;

import ai.skymind.nativerl.PolicyHelper;
import java.io.IOException;
import java.io.File;
import java.io.FilenameFilter;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import org.bytedeco.javacpp.Loader;
import org.junit.rules.TemporaryFolder;
import org.junit.Rule;
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
public class ModelTest {
    @Rule public TemporaryFolder folder = new TemporaryFolder();

    void execute(File directory, String... command) throws Exception {
        assertEquals(0, new ProcessBuilder(command).directory(directory).inheritIO().start().waitFor());
    }

    File[] find(File root, String filename) {
        ArrayList<File> files = new ArrayList<File>();
        FilenameFilter filter = (File dir, String name) -> {
            File f = new File(dir, name);
            if (name.equals(filename)) {
                files.add(f);
            }
            return f.isDirectory();
        };

        ArrayList<File> dirs = new ArrayList<File>(Arrays.asList(root.listFiles(filter)));
        while (!dirs.isEmpty()) {
            File d = dirs.remove(dirs.size() - 1);
            File[] dirs2 = d.listFiles(filter);
            dirs.addAll(Arrays.asList(dirs2));
        }

        return files.toArray(new File[files.size()]);
    }

    @Test public void testTrafficPhases() throws Exception {
        File binDir = new File("target/dependency/nativerl-bin/");
        File exportDir = folder.newFolder("TrafficPhases");
        File modelDir = new File(getClass().getResource("trafficphases").toURI());
        File simulationDir = new File(exportDir, "TrafficPhases_Simulation");

        execute(modelDir, "anylogic", "-e", "-o", exportDir.getAbsolutePath(), modelDir.getAbsolutePath() + "/TrafficPhases.alp");

        for (File f : binDir.listFiles()) {
            Files.copy(f.toPath(), simulationDir.toPath().resolve(f.getName()));
        }
        Files.copy(new File(binDir, "examples/traintraffic.sh").toPath(), simulationDir.toPath().resolve("traintraffic.sh"));
        execute(simulationDir, "bash", "traintraffic.sh");

        File[] savedModels = find(simulationDir, "saved_model.pb");
        assertTrue(savedModels.length > 0);
        for (File f : savedModels) {
            PolicyHelper h = PolicyHelper.load(f.getParentFile());
            double[] o = h.computeActions(new double[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
            System.out.println(Arrays.toString(o));
            assertEquals(o.length, 1);
            assertThat(o[0], anyOf(is(0.0), is(1.0)));
        }
    }
}
