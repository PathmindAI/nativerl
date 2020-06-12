package ai.skymind.nativerl;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;

import com.github.jknack.handlebars.Handlebars;
import com.github.jknack.handlebars.Helper;
import com.github.jknack.handlebars.Options;
import com.github.jknack.handlebars.Template;
import com.github.jknack.handlebars.helper.ConditionalHelpers;
import com.github.jknack.handlebars.io.ClassPathTemplateLoader;
import com.github.jknack.handlebars.io.TemplateLoader;
import lombok.Getter;
import org.bytedeco.cpython.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.numpy.*;
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
    String algorithm = "PPO";
    File outputDir = null;
    File checkpoint = null;
    Environment environment = null;
    int numGPUs = 0;
    int numWorkers = 1;
    int numHiddenLayers = 2;
    int numHiddenNodes = 256;
    int maxIterations = 500;
    int maxTimeInSec = -1;
    int numSamples = 4;
    int savePolicyInterval = 100;
    String redisAddress = null;
    String customParameters = "";
    boolean resume = false;
    int checkpointFrequency = 50;
    boolean multiAgent = false;
    boolean userLog = false;

    // thresholds for stopper
    double episodeRewardRangeTh = 0.01; // episode_reward_range_threshold
    double entropySlopeTh = 0.01;       // entropy_slope_threshold
    double vfLossRangeTh = 0.1;         // vf_loss_range_threshold
    double valuePredTh = 0.01;          // value_pred_threshold

    public RLlibHelper() {
    }

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

    public File[] rllibpaths() {
        return rllibpaths;
    }
    public RLlibHelper rllibpaths(File[] rllibpaths) {
        this.rllibpaths = rllibpaths;
        return this;
    }
    public RLlibHelper rllibpaths(String[] rllibpaths) {
        File[] files = new File[rllibpaths.length];
        for (int i = 0; i < files.length; i++) {
            files[i] = new File(rllibpaths[i]);
        }
        this.rllibpaths = files;
        return this;
    }

    public String algorithm() {
        return algorithm;
    }
    public RLlibHelper algorithm(String algorithm) {
        this.algorithm = algorithm;
        return this;
    }

    public File outputDir() {
        return outputDir;
    }
    public RLlibHelper outputDir(File outputDir) {
        this.outputDir = outputDir;
        return this;
    }
    public RLlibHelper outputDir(String outputDir) {
        this.outputDir = new File(outputDir);
        return this;
    }

    public File checkpoint() {
        return checkpoint;
    }
    public RLlibHelper checkpoint(File checkpoint) {
        this.checkpoint = checkpoint;
        return this;
    }
    public RLlibHelper checkpoint(String checkpoint) {
        this.checkpoint = new File(checkpoint);
        return this;
    }

    public Environment environment() {
        return environment;
    }
    public RLlibHelper environment(Environment environment) {
        this.environment = environment;
        return this;
    }

    public int numGPUs() {
        return numGPUs;
    }
    public RLlibHelper numGPUs(int numGPUs) {
        this.numGPUs = numGPUs;
        return this;
    }

    public int numWorkers() {
        return numWorkers;
    }
    public RLlibHelper numWorkers(int numWorkers) {
        this.numWorkers = numWorkers;
        return this;
    }

    public int numHiddenLayers() {
        return numHiddenLayers;
    }
    public RLlibHelper numHiddenLayers(int numHiddenLayers) {
        this.numHiddenLayers = numHiddenLayers;
        return this;
    }

    public int numHiddenNodes() {
        return numHiddenNodes;
    }
    public RLlibHelper numHiddenNodes(int numHiddenNodes) {
        this.numHiddenNodes = numHiddenNodes;
        return this;
    }

    public int maxIterations() {
        return maxIterations;
    }
    public RLlibHelper maxIterations(int maxIterations) {
        this.maxIterations = maxIterations;
        return this;
    }

    public int maxTimeInSec() {
        return maxTimeInSec;
    }

    public RLlibHelper maxTimeInSec(int maxTimeInSec) {
        this.maxTimeInSec = maxTimeInSec;
        return this;
    }

    public int numSamples() {
        return numSamples;
    }

    public RLlibHelper numSamples(int numSamples) {
        this.numSamples = numSamples;
        return this;
    }

    public boolean resume() {
        return resume;
    }

    public RLlibHelper resume(boolean resume) {
        this.resume = resume;
        return this;
    }

    public int checkpointFrequency() {
        return checkpointFrequency;
    }

    public RLlibHelper checkpointFrequency(int checkpointFrequency) {
        this.checkpointFrequency = checkpointFrequency;
        return this;
    }

    public int savePolicyInterval() {
        return savePolicyInterval;
    }
    public RLlibHelper savePolicyInterval(int savePolicyInterval) {
        this.savePolicyInterval = savePolicyInterval;
        return this;
    }

    public String redisAddress() {
        return redisAddress;
    }
    public RLlibHelper redisAddress(String redisAddress) {
        this.redisAddress = redisAddress;
        return this;
    }

    public String customParameters() {
        return customParameters;
    }
    public RLlibHelper customParameters(String customParameters) {
        this.customParameters = customParameters;
        return this;
    }

    public boolean isMultiAgent() {
        return multiAgent;
    }
    public RLlibHelper setMultiAgent(boolean multiAgent) {
        this.multiAgent = multiAgent;
        return this;
    }

    public double episodeRewardRangeTh() {
        return episodeRewardRangeTh;
    }

    public RLlibHelper episodeRewardRangeTh(double episodeRewardRangeTh) {
        this.episodeRewardRangeTh = episodeRewardRangeTh;
        return this;
    }

    public double entropySlopeTh() {
        return entropySlopeTh;
    }

    public RLlibHelper entropySlopeTh(double entropySlopeTh) {
        this.entropySlopeTh = entropySlopeTh;
        return this;
    }

    public double vfLossRangeTh() {
        return vfLossRangeTh;
    }

    public RLlibHelper vfLossRangeTh(double vfLossRangeTh) {
        this.vfLossRangeTh = vfLossRangeTh;
        return this;
    }

    public double valuePredTh() {
        return valuePredTh;
    }

    public RLlibHelper valuePredTh(double valuePredTh) {
        this.valuePredTh = valuePredTh;
        return this;
    }

    public boolean userLog() {
        return userLog;
    }

    public RLlibHelper userLog(boolean userLog) {
        this.userLog = userLog;
        return this;
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
        RLlibHelper helper = new RLlibHelper();
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
                helper.rllibpaths(args[++i].split(File.pathSeparator));
            } else if ("--algorithm".equals(args[i])) {
                helper.algorithm(args[++i]);
            } else if ("--output-dir".equals(args[i])) {
                helper.outputDir(args[++i]);
            } else if ("--checkpoint".equals(args[i])) {
                helper.checkpoint(args[++i]);
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
                helper.resume = true;
            } else if ("--checkpoint-frequency".equals(args[i])) {
                helper.checkpointFrequency(Integer.parseInt(args[++i]));
            } else if ("--multi-agent".equals(args[i])) {
                helper.multiAgent = true;
            } else if ("--episode-reward-range".equals(args[i])) {
                helper.episodeRewardRangeTh(Double.parseDouble(args[++i]));
            } else if ("--entropy-slope".equals(args[i])) {
                helper.entropySlopeTh(Double.parseDouble(args[++i]));
            } else if ("--vf-loss-range".equals(args[i])) {
                helper.vfLossRangeTh(Double.parseDouble(args[++i]));
            } else if ("--value-pred".equals(args[i])) {
                helper.valuePredTh(Double.parseDouble(args[++i]));
            } else if ("--user-log".equals(args[i])) {
                helper.userLog = true;
            } else {
                output = new File(args[i]);
            }
        }
        helper.generatePythonTrainer(output);
    }
}
