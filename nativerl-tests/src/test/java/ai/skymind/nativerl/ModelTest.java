package ai.skymind.nativerl;

import java.io.File;
import java.io.FilenameFilter;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;
import org.junit.Rule;
import org.junit.rules.TemporaryFolder;

import static org.junit.Assert.assertEquals;

/**
 *
 * @author saudet
 */
public class ModelTest {
    @Rule public TemporaryFolder folder = new TemporaryFolder();

    public static void copy(File src, File dst) throws Exception {
        if (src.isDirectory()) {
            for (File f : src.listFiles()) {
                Files.copy(f.toPath(), dst.toPath().resolve(f.getName()), StandardCopyOption.REPLACE_EXISTING);
            }
        } else {
            Files.copy(src.toPath(), dst.toPath().resolve(src.getName()), StandardCopyOption.REPLACE_EXISTING);
        }
    }

    public static void execute(File directory, String... command) throws Exception {
        execute(directory, null, command);
    }
    public static void execute(File directory, Map<String, String> environment, String... command) throws Exception {
        ProcessBuilder pb = new ProcessBuilder(command).directory(directory).inheritIO();
        if (environment != null) {
            pb.environment().putAll(environment);
        }
        assertEquals(0, pb.start().waitFor());
    }

    public static File[] find(File root, String filename) {
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
            dirs.addAll(Arrays.asList(d.listFiles(filter)));
        }

        return files.toArray(new File[files.size()]);
    }
}
