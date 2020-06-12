package ai.skymind.nativerl;

import com.github.jknack.handlebars.Handlebars;
import com.github.jknack.handlebars.Template;
import com.github.jknack.handlebars.helper.ConditionalHelpers;
import com.github.jknack.handlebars.io.ClassPathTemplateLoader;
import com.github.jknack.handlebars.io.TemplateLoader;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.bytedeco.cpython.PyObject;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.SizeTPointer;
import org.bytedeco.numpy.PyArrayObject;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;

import static org.bytedeco.cpython.global.python.*;
import static org.bytedeco.numpy.global.numpy.*;

/**
 * Currently available algorithms according to RLlib's registry.py:
 *   "DDPG" *
 *   "APEX_DDPG" *
 *   "TD3" *
 *   "PPO"
 *   "ES"
 *   "ARS"
 *   "DQN"
 *   "APEX"
 *   "A3C"
 *   "A2C"
 *   "PG"
 *   "IMPALA"
 *   "QMIX" **
 *   "APEX_QMIX" **
 *   "APPO"
 *   "MARWIL"
 *
 *   * Works only with continuous actions (doesn't work with discrete ones)
 *   ** Requires PyTorch (doesn't work with TensorFlow)
 */
@Getter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class RLlibHelper {

    public static class PythonPolicyHelper implements PolicyHelper {
        PyObject globals = null;
        PyArrayObject obsArray = null;
        FloatPointer obsData = null;

        public PythonPolicyHelper(File[] rllibpaths, String algorithm, File checkpoint, Environment env) throws IOException {
            this(rllibpaths, algorithm, checkpoint, env.getClass().getSimpleName(), env.getActionSpace(), env.getObservationSpace());
        }
        public PythonPolicyHelper(File[] rllibpaths, String algorithm, File checkpoint, String name, long discreteActions, long continuousObservations) throws IOException {
            this(rllibpaths, algorithm, checkpoint, name, AbstractEnvironment.getDiscreteSpace(discreteActions),
                    AbstractEnvironment.getContinuousSpace(continuousObservations));
        }
        public PythonPolicyHelper(File[] rllibpaths, String algorithm, File checkpoint, String name, Space actionSpace, Space obsSpace) throws IOException {
            File[] paths = org.bytedeco.numpy.global.numpy.cachePackages();
            paths = Arrays.copyOf(paths, paths.length + rllibpaths.length);
            System.arraycopy(rllibpaths, 0, paths, paths.length - rllibpaths.length, rllibpaths.length);
            Py_SetPath(paths);
            Pointer program = Py_DecodeLocale(name, null);
            Py_SetProgramName(program);
            Py_Initialize();
            PySys_SetArgvEx(1, program, 0);
            if (_import_array() < 0) {
                PyErr_Print();
                PyErr_Clear();
                throw new RuntimeException("numpy.core.multiarray failed to import");
            }
            PyObject module = PyImport_AddModule("__main__");
            globals = PyModule_GetDict(module);

            Discrete discreteActionSpace = (Discrete)actionSpace;
            Continuous continuousObsSpace = (Continuous)obsSpace;
            float[] obsLow = continuousObsSpace.low().get();
            float[] obsHigh = continuousObsSpace.high().get();
            long[] obsShape = continuousObsSpace.shape().get();

            PyRun_StringFlags("import gym, inspect, ray, sys\n"
                    + "import numpy as np\n"
                    + "from ray.rllib.agents import registry\n"
                    + "\n"
                    + "class " + name + "(gym.Env):\n"
                    + "    def __init__(self, env_config):\n"
                    + "        self.action_space = gym.spaces.Discrete(" + discreteActionSpace.n() + ")\n"
                    + "        low = " + (obsLow.length == 1 ? obsLow[0] : "np.array(" + Arrays.toString(obsLow) + ")") + "\n"
                    + "        high = " + (obsHigh.length == 1 ? obsHigh[0] : "np.array(" + Arrays.toString(obsHigh) + ")") + "\n"
                    + "        shape = " + (obsShape.length == 0 ? "None" : "np.array(" + Arrays.toString(obsShape) + ")") + "\n"
                    + "        self.observation_space = gym.spaces.Box(low, high, shape=shape, dtype=np.float32)\n"
                    + "\n"
                    + "ray.init(local_mode=True)\n"
                    + "cls = registry.get_agent_class(\"" + algorithm + "\")\n"
                    + "config = inspect.getmodule(cls).DEFAULT_CONFIG.copy()\n"
                    + "config[\"num_gpus\"] = 0\n"
                    + "config[\"num_workers\"] = 1\n"
                    + "trainer = cls(config=config, env=" + name + ")\n"
                    + "\n"
                    + "# Can optionally call trainer.restore(path) to load a checkpoint.\n"
                    + "trainer.restore(\"" + checkpoint.getAbsolutePath() + "\")\n", Py_file_input, globals, globals, null);

            if (PyErr_Occurred() != null) {
                PyErr_Print();
                PyErr_Clear();
                PyRun_StringFlags("sys.stderr.flush()", Py_file_input, globals, globals, null);
                throw new RuntimeException("Python error occurred");
            }

            if (obsShape.length == 0) {
                obsShape = new long[] {obsLow.length};
            }
            obsArray = new PyArrayObject(PyArray_New(PyArray_Type(), obsShape.length, new SizeTPointer(obsShape),
                                                     NPY_FLOAT, null, null, 0, 0, null));
            obsData = new FloatPointer(PyArray_BYTES(obsArray)).capacity(PyArray_Size(obsArray));
            PyDict_SetItemString(globals, "obs", obsArray);
        }

        @Override public float[] computeContinuousAction(float[] state) {
            throw new UnsupportedOperationException();
        }

        @Override public long computeDiscreteAction(float[] state) {
            obsData.put(state);
            PyRun_StringFlags("action = trainer.compute_action(obs)\n", Py_file_input, globals, globals, null);

            if (PyErr_Occurred() != null) {
                PyErr_Print();
                PyErr_Clear();
                PyRun_StringFlags("sys.stderr.flush()", Py_file_input, globals, globals, null);
                throw new RuntimeException("Python error occurred");
            }
            return PyLong_AsLongLong(PyDict_GetItemString(globals, "action"));
        }
    }

    File[] rllibpaths = null;
    File outputDir = null;
    File checkpoint = null;
    Environment environment = null;
    String redisAddress = null;
    boolean resume = false;
    boolean multiAgent = false;
    boolean userLog = false;

    int numGPUs = 0;

    @Builder.Default
    String algorithm = "PPO";

    @Builder.Default
    int numWorkers = 1;

    @Builder.Default
    int numHiddenLayers = 2;

    @Builder.Default
    int numHiddenNodes = 256;

    @Builder.Default
    int maxIterations = 500;

    @Builder.Default
    int maxTimeInSec = -1;

    @Builder.Default
    int numSamples = 4;

    @Builder.Default
    int savePolicyInterval = 100;

    @Builder.Default
    String customParameters = "";

    @Builder.Default
    int checkpointFrequency = 50;

    // thresholds for stopper
    @Builder.Default
    double episodeRewardRangeTh = 0.01; // episode_reward_range_threshold

    @Builder.Default
    double entropySlopeTh = 0.01;       // entropy_slope_threshold

    @Builder.Default
    double vfLossRangeTh = 0.1;         // vf_loss_range_threshold

    @Builder.Default
    double valuePredTh = 0.01;          // value_pred_threshold

    public RLlibHelper(RLlibHelper copy) {
        this.rllibpaths = copy.rllibpaths;
        this.algorithm = copy.algorithm;
        this.outputDir = copy.outputDir;
        this.checkpoint = copy.checkpoint;
        this.environment = copy.environment;
        this.numGPUs = copy.numGPUs;
        this.numWorkers = copy.numWorkers;
        this.numHiddenLayers = copy.numHiddenLayers;
        this.numHiddenNodes = copy.numHiddenNodes;
        this.maxIterations = copy.maxIterations;
        this.savePolicyInterval = copy.savePolicyInterval;
        this.redisAddress = copy.redisAddress;
        this.maxTimeInSec = copy.maxTimeInSec;
        this.customParameters = copy.customParameters;
        this.numSamples = copy.numSamples;
        this.resume = copy.resume;
        this.checkpointFrequency = copy.checkpointFrequency;
        this.userLog = copy.userLog;
        this.episodeRewardRangeTh = copy.episodeRewardRangeTh;
        this.entropySlopeTh = copy.entropySlopeTh;
        this.vfLossRangeTh = copy.vfLossRangeTh;
        this.valuePredTh = copy.valuePredTh;
    }

    @Override public String toString() {
        return "RLlibHelper[rllibpaths=" + Arrays.deepToString(rllibpaths) + ", "
                + "algorithm=" + algorithm + ", "
                + "outputDir=" + outputDir + ", "
                + "checkpoint=" + checkpoint + ", "
                + "environment=" + environment + ", "
                + "numGPUs=" + numGPUs + ", "
                + "numWorkers=" + numWorkers + ", "
                + "numHiddenLayers=" + numHiddenLayers + ", "
                + "numHiddenNodes=" + numHiddenNodes  + ", "
                + "maxIterations=" + maxIterations + ", "
                + "savePolicyInterval=" + savePolicyInterval + ", "
                + "maxTimeInSec=" + maxTimeInSec + ", "
                + "redisAddress=" + redisAddress + ", "
                + "resume=" + resume + ", "
                + "checkpointFrequency=" + checkpointFrequency + ", "
                + "episodeRewardRangeTh=" + episodeRewardRangeTh + ", "
                + "entropySlopeTh=" + entropySlopeTh + ", "
                + "entropySlopeTh=" + entropySlopeTh + ", "
                + "vfLossRangeTh=" + vfLossRangeTh + ", "
                + "userLog=" + userLog + ", "
                + "customParameters=" + customParameters + "]";
    }

    public String hiddenLayers() {
        String s = "[";
        for (int i = 0; i < numHiddenLayers; i++) {
            s += numHiddenNodes + (i < numHiddenLayers - 1 ? ", " : "]");
        }
        return s;
    }

    public String getHiddenLayers() {
        return hiddenLayers();
    }

    public PolicyHelper createPythonPolicyHelper() throws IOException {
        return new PythonPolicyHelper(rllibpaths, algorithm, checkpoint, environment);
    }

    public void generatePythonTrainer(File file) throws IOException {
        File directory = file.getParentFile();
        if (directory != null) {
            directory.mkdirs();
        }
        Files.write(file.toPath(), generatePythonTrainer().getBytes());
    }

    public String generatePythonTrainer() throws IOException {
        if (environment == null) {
            throw new IllegalStateException("Environment is null.");
        }
        TemplateLoader loader = new ClassPathTemplateLoader();
        loader.setSuffix(".hbs");
        Handlebars handlebars = new Handlebars(loader);

        handlebars.registerHelpers(ConditionalHelpers.class);
        handlebars.registerHelper("className", (context, options) -> context.getClass().getName());
        handlebars.registerHelper("classSimpleName", (context, options) -> context.getClass().getSimpleName());

        Template template = handlebars.compile("RLlibHelper.py");

        String trainer = template.apply(this);
        return trainer;
    }

    public static void main(String[] args) throws Exception {
        RLlibHelper.RLlibHelperBuilder helper = RLlibHelper.builder();
        File output = new File("rllibtrain.py");
        for (int i = 0; i < args.length; i++) {
            if ("-help".equals(args[i]) || "--help".equals(args[i])) {
                System.out.println("usage: RLlibHelper [options] [output]");
                System.out.println();
                System.out.println("options:");
                System.out.println("    --rllibpaths");
                System.out.println("    --algorithm");
                System.out.println("    --checkpoint");
                System.out.println("    --environment");
                System.out.println("    --num-gpus");
                System.out.println("    --num-workers");
                System.out.println("    --num-hidden-layers");
                System.out.println("    --num-hidden-nodes");
                System.out.println("    --max-iterations");
                System.out.println("    --save-policy-interval");
                System.out.println("    --redis-address");
                System.out.println("    --custom-parameters");
                System.out.println("    --multi-agent");
                System.out.println("    --maxTimeInSec");
                System.out.println("    --num-samples");
                System.out.println("    --resume");
                System.out.println("    --checkpoint-frequency");
                System.out.println("    --episode-reward-range");
                System.out.println("    --entropy-slope");
                System.out.println("    --vf-loss-range");
                System.out.println("    --value-pred");
                System.out.println("    --user-log");
                System.exit(0);
            } else if ("--rllibpaths".equals(args[i])) {
                helper.rllibpaths(rllibpaths(args[++i].split(File.pathSeparator)));
            } else if ("--algorithm".equals(args[i])) {
                helper.algorithm(args[++i]);
            } else if ("--output-dir".equals(args[i])) {
                helper.outputDir(new File(args[++i]));
            } else if ("--checkpoint".equals(args[i])) {
                helper.checkpoint(new File(args[++i]));
            } else if ("--environment".equals(args[i])) {
                helper.environment(Class.forName(args[++i]).asSubclass(Environment.class).newInstance());
            } else if ("--num-gpus".equals(args[i])) {
                helper.numGPUs(Integer.parseInt(args[++i]));
            } else if ("--num-workers".equals(args[i])) {
                helper.numWorkers(Integer.parseInt(args[++i]));
            } else if ("--num-hidden-layers".equals(args[i])) {
                helper.numHiddenLayers(Integer.parseInt(args[++i]));
            } else if ("--num-hidden-nodes".equals(args[i])) {
                helper.numHiddenNodes(Integer.parseInt(args[++i]));
            } else if ("--max-iterations".equals(args[i])) {
                helper.maxIterations(Integer.parseInt(args[++i]));
            } else if ("--max-time-in-sec".equals(args[i])) {
                helper.maxTimeInSec(Integer.parseInt(args[++i]));
            } else if ("--num-samples".equals(args[i])) {
                helper.numSamples(Integer.parseInt(args[++i]));
            } else if ("--save-policy-interval".equals(args[i])) {
                helper.savePolicyInterval(Integer.parseInt(args[++i]));
            } else if ("--redis-address".equals(args[i])) {
                helper.redisAddress(args[++i]);
            } else if ("--custom-parameters".equals(args[i])) {
                helper.customParameters(args[++i]);
            } else if ("--resume".equals(args[i])) {
                helper.resume(true);
            } else if ("--checkpoint-frequency".equals(args[i])) {
                helper.checkpointFrequency(Integer.parseInt(args[++i]));
            } else if ("--multi-agent".equals(args[i])) {
                helper.multiAgent(true);
            } else if ("--episode-reward-range".equals(args[i])) {
                helper.episodeRewardRangeTh(Double.parseDouble(args[++i]));
            } else if ("--entropy-slope".equals(args[i])) {
                helper.entropySlopeTh(Double.parseDouble(args[++i]));
            } else if ("--vf-loss-range".equals(args[i])) {
                helper.vfLossRangeTh(Double.parseDouble(args[++i]));
            } else if ("--value-pred".equals(args[i])) {
                helper.valuePredTh(Double.parseDouble(args[++i]));
            } else if ("--user-log".equals(args[i])) {
                helper.userLog(true);
            } else {
                output = new File(args[i]);
            }
        }
        helper.build().generatePythonTrainer(output);
    }

    public static File[] rllibpaths(String[] rllibpaths) {
        File[] files = new File[rllibpaths.length];
        for (int i = 0; i < files.length; i++) {
            files[i] = new File(rllibpaths[i]);
        }
        return files;
    }

}
