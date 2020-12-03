package ai.skymind.nativerl;

import ai.skymind.nativerl.util.AutoregressiveModelHelper;
import com.github.jknack.handlebars.Handlebars;
import com.github.jknack.handlebars.Template;
import com.github.jknack.handlebars.helper.ConditionalHelpers;
import com.github.jknack.handlebars.io.ClassPathTemplateLoader;
import com.github.jknack.handlebars.io.TemplateLoader;
import lombok.*;
import org.bytedeco.cpython.PyObject;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.SizeTPointer;
import org.bytedeco.numpy.PyArrayObject;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.Collections;

import static org.bytedeco.cpython.global.python.*;
import static org.bytedeco.numpy.global.numpy.*;

/**
* This is a helper class to help users use an implementation of
* the reinforcement learning Environment interface using RLlib.
* The output is a Python script that can executed with an existing
* installation of RLlib.
* <p>
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
 * <p>
 *   * Works only with continuous actions (doesn't work with discrete ones)
 *   ** Requires PyTorch (doesn't work with TensorFlow)
 * <p>
 */
@Getter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class RLlibHelper {

    /**
     * A PolicyHelper for RLlib, which can load its checkpoint files.
     * Requires CPython and comes with all its limitations, such as the GIL.
     * For example:
     <pre>{@code
         // Directories where to find all modules required by RLlib
         File[] path = {
             new File("/usr/lib64/python3.7/lib-dynload/"),
             new File("/usr/lib64/python3.7/site-packages/"),
             new File("/usr/lib/python3.7/site-packages/"),
             new File(System.getProperty("user.home") + "/.local/lib/python3.7/site-packages/")
         };
         File checkpoint = new File("/path/to/checkpoint_100/checkpoint-100");
         PolicyHelper policyHelper = new RLlibHelper.PythonPolicyHelper(path, "PPO", checkpoint, "Traffic", 2, 10);
         int action = (int)policyHelper.computeDiscreteAction(getObservation(false));
     }</pre>
     *
     */
    public static class PythonPolicyHelper implements PolicyHelper {
        PyObject globals = null;
        PyArrayObject obsArray = null;
        FloatPointer obsData = null;
        int actionTupleSize;

        public PythonPolicyHelper(File[] rllibpaths, String algorithm, File checkpoint, String environment) throws IOException, ReflectiveOperationException {
            this(rllibpaths, algorithm, checkpoint, environment, NativeRL.createEnvironment(environment).getActionSpace(0), NativeRL.createEnvironment(environment).getObservationSpace());
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

            actionTupleSize = (int)discreteActionSpace.size();
            String actionSpaceStr = "gym.spaces.Discrete(" + discreteActionSpace.n() + ")";
            PyRun_StringFlags("import gym, inspect, numpy, ray, sys\n"
                    + "from ray.rllib.agents import registry\n"
                    + "\n"
                    + "class " + name + "(gym.Env):\n"
                    + "    def __init__(self, env_config):\n"
                    + "        self.action_space = gym.spaces.Tuple([" + String.join(",", Collections.nCopies(actionTupleSize, actionSpaceStr)) + " ])\n"
                    + "        low = " + (obsLow.length == 1 ? obsLow[0] : "numpy.array(" + Arrays.toString(obsLow) + ")") + "\n"
                    + "        high = " + (obsHigh.length == 1 ? obsHigh[0] : "numpy.array(" + Arrays.toString(obsHigh) + ")") + "\n"
                    + "        shape = " + (obsShape.length == 0 ? "None" : "numpy.array(" + Arrays.toString(obsShape) + ")") + "\n"
                    + "        self.observation_space = gym.spaces.Box(low, high, shape=shape, dtype=numpy.float32)\n"
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

        @Override public float[] computeActions(float[] state) {
            obsData.put(state);
            PyRun_StringFlags("action = trainer.compute_action(obs).astype(float)\n", Py_file_input, globals, globals, null);

            if (PyErr_Occurred() != null) {
                PyErr_Print();
                PyErr_Clear();
                PyRun_StringFlags("sys.stderr.flush()", Py_file_input, globals, globals, null);
                throw new RuntimeException("Python error occurred");
            }

            float[] arrayOfActions = new float[actionTupleSize];
            for (int i=0; i < actionTupleSize; i++) {
                arrayOfActions[i] = (float)PyFloat_AsDouble(PyDict_GetItemString(globals, "action[i]"));
            }
            return arrayOfActions;
            //return PyLong_AsLongLong(PyDict_GetItemString(globals, "action",));
        }

        @Override public long[] computeDiscreteAction(float[] state) {
            obsData.put(state);
            PyRun_StringFlags("action = trainer.compute_action(obs)\n", Py_file_input, globals, globals, null);

            if (PyErr_Occurred() != null) {
                PyErr_Print();
                PyErr_Clear();
                PyRun_StringFlags("sys.stderr.flush()", Py_file_input, globals, globals, null);
                throw new RuntimeException("Python error occurred");
            }

            long[] arrayOfActions = new long[actionTupleSize];
            for (int i=0; i < actionTupleSize; i++) {
                arrayOfActions[i] = PyLong_AsLongLong(PyDict_GetItemString(globals, "action[i]"));
            }
            return arrayOfActions;
            //return PyLong_AsLongLong(PyDict_GetItemString(globals, "action",));
        }
    }

    /** The paths where to find RLlib itself and all of its Python dependencies. */
    @Builder.Default
    File[] rllibpaths = null;

    /** The algorithm to use with RLlib for training and the PythonPolicyHelper. */
    @Builder.Default
    String algorithm = "PPO";

    /** The directory where to output the logs of RLlib. */
    @Builder.Default
    File outputDir = null;

    /** The RLlib checkpoint to restore for the PythonPolicyHelper or to start training from instead of a random policy. */
    @Builder.Default
    File checkpoint = null;

    /** The name of a subclass of Environment to use as environment for training and/or with PythonPolicyHelper. */
    @Builder.Default
    String environment = null;

    /** The maximum amount of memory in MB to use for Java environments (passed via the "-Xmx" argument). */
    @Builder.Default
    int maxMemoryInMB = 4096;

    /** The number of CPU cores to let RLlib use during training. */
    @Builder.Default
    int numCPUs = 1;

    /** The number of GPUs to let RLlib use during training. */
    @Builder.Default
    int numGPUs = 0;

    /** The number of parallel workers that RLlib should execute during training. */
    @Builder.Default
    int numWorkers = 1;

    /** The number of hidden layers in the MLP to use for the learning model. */
    @Builder.Default
    int numHiddenLayers = 2;

    /** The number of nodes per layer in the MLP to use for the learning model. */
    @Builder.Default
    int numHiddenNodes = 256;

    /** The maximum number of training iterations as a stopping criterion. */
    @Builder.Default
    int maxIterations = 500;

    /** Max time in seconds */
    @Builder.Default
    int maxTimeInSec = 43200;

    /** Number of population-based training samples */
    @Builder.Default
    int numSamples = 4;

    /** Length of actions array for tuples */
    @Builder.Default
    int actionTupleSize = 1;

    /** The frequency at which policies should be saved to files, given as an interval in the number of training iterations. */
    @Builder.Default
    int savePolicyInterval = 100;

    /** Initialize actions as a long */
//    @Builder.Default
    long discreteActions;

    /** The address of the Redis server for distributed training sessions. */
    @Builder.Default
    String redisAddress = null;

    /** Any number custom parameters written in Python appended to the config of ray.tune.run() as is. */
    @Builder.Default
    String customParameters = "";

    /** Resume training when AWS spot instance terminates */
    @Builder.Default
    boolean resume = false;

    /** Periodic checkpointing to allow training to recover from AWS spot instance termination */
    @Builder.Default
    int checkpointFrequency = 50;

    /** Indicates that we need multiagent support with the Environment class provided, but where all agents share the same policy. */
    @Builder.Default
    boolean multiAgent = false;

    /** Indicates that we save a raw metrics tata to metrics_raw column in progress.csv*/
    @Builder.Default
    boolean debugMetrics = false;

    /** Reduce size of output log file */
    @Builder.Default
    boolean userLog = false;

    /** Optional layer on top of tuples */
    @Builder.Default
    boolean autoregressive = false;

    @Setter
    String autoregressiveModel;

    // Thresholds for stopper

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
        this.maxMemoryInMB = copy.maxMemoryInMB;
        this.numCPUs = copy.numCPUs;
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
        this.actionTupleSize = copy.actionTupleSize;
        this.autoregressive = copy.autoregressive;
        this.discreteActions = copy.discreteActions;
        this.debugMetrics = copy.debugMetrics;
    }

    @Override public String toString() {
        return "RLlibHelper[rllibpaths=" + Arrays.deepToString(rllibpaths) + ", "
                + "algorithm=" + algorithm + ", "
                + "outputDir=" + outputDir + ", "
                + "actionTupleSize=" + actionTupleSize + ", "
                + "autoregressive=" + autoregressive + ", "
                + "checkpoint=" + checkpoint + ", "
                + "maxMemoryInMB=" + maxMemoryInMB + ", "
                + "environment=" + environment + ", "
                + "numCPUs=" + numCPUs + ", "
                + "numGPUs=" + numGPUs + ", "
                + "numWorkers=" + numWorkers + ", "
                + "numSamples=" + numSamples + ", "
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
                + "debugMetrics=" + debugMetrics + ", "
                + "discreteActions=" + discreteActions + ", "
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

    public PolicyHelper createPythonPolicyHelper() throws IOException, ReflectiveOperationException {
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
        TemplateLoader loader = new ClassPathTemplateLoader("/ai/skymind/nativerl", ".hbs");
        Handlebars handlebars = new Handlebars(loader);

        handlebars.registerHelpers(ConditionalHelpers.class);
        handlebars.registerHelper("className", (context, options) -> context);
        handlebars.registerHelper("classSimpleName", (context, options) -> ((String)context).substring(((String)context).lastIndexOf('.') + 1));
        handlebars.registerHelper("escapePath", (context, options) -> ((File)context).getAbsolutePath().replace("\\", "/"));

        Template template = handlebars.compile("RLlibHelper.py");

        if (autoregressive) {
            setAutoregressiveModel(AutoregressiveModelHelper.generateAutoregressiveClass(actionTupleSize, discreteActions));
        }

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
                System.out.println("    --max-memory-in-mb");
                System.out.println("    --num-cpus");
                System.out.println("    --num-gpus");
                System.out.println("    --num-workers");
                System.out.println("    --num-hidden-layers");
                System.out.println("    --num-hidden-nodes");
                System.out.println("    --max-iterations");
                System.out.println("    --save-policy-interval");
                System.out.println("    --redis-address");
                System.out.println("    --custom-parameters");
                System.out.println("    --multi-agent");
                System.out.println("    --max-time-in-sec");
                System.out.println("    --num-samples");
                System.out.println("    --resume");
                System.out.println("    --checkpoint-frequency");
                System.out.println("    --episode-reward-range");
                System.out.println("    --entropy-slope");
                System.out.println("    --vf-loss-range");
                System.out.println("    --value-pred");
                System.out.println("    --user-log");
                System.out.println("    --debug-metrics");
                System.out.println("    --action-tuple-size");
                System.out.println("    --autoregressive");
                System.out.println("    --discrete-actions");
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
                helper.environment(args[++i]);
            } else if ("--max-memory-in-mb".equals(args[i])) {
                helper.maxMemoryInMB(Integer.parseInt(args[++i]));
            } else if ("--num-cpus".equals(args[i])) {
                helper.numCPUs(Integer.parseInt(args[++i]));
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
            } else if ("--debug-metrics".equals(args[i])) {
                helper.debugMetrics(true);
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
            } else if ("--autoregressive".equals(args[i])) {
                helper.autoregressive(true);
            } else if ("--action-tuple-size".equals(args[i])) {
                helper.actionTupleSize(Integer.parseInt(args[++i]));
            } else if ("--discrete-actions".equals(args[i])) {
                helper.discreteActions(Long.parseLong(args[++i]));
            } else if (args[i].endsWith(".py")) {
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
