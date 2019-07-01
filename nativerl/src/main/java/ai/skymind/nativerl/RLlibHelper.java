package ai.skymind.nativerl;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;
import org.bytedeco.cpython.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;
import org.bytedeco.numpy.*;
import static org.bytedeco.cpython.global.python.*;
import static org.bytedeco.numpy.global.numpy.*;

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

            PyRun_StringFlags("import gym, inspect, numpy, ray, sys\n"
                    + "from ray.rllib.agents import registry\n"
                    + "\n"
                    + "class " + name + "(gym.Env):\n"
                    + "    def __init__(self, env_config):\n"
                    + "        self.action_space = gym.spaces.Discrete(" + discreteActionSpace.n() + ")\n"
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
    File checkpoint = null;
    Environment environment = null;
    int numGPUs = 0;
    int numWorkers = 1;
    long randomSeed = 0;
    double[] gammas = {0.99};
    double[] learningRates = {5e-5};
    int[] miniBatchSizes = {128};
    int numHiddenLayers = 2;
    int numHiddenNodes = 256;
    int stepsPerIteration = 4000;
    int maxIterations = 500;
    int savePolicyInterval = 100;
    String redisAddress = null;

    File[] rllibpaths() {
        return rllibpaths;
    }
    RLlibHelper rllibpaths(File[] rllibpaths) {
        this.rllibpaths = rllibpaths;
        return this;
    }
    RLlibHelper rllibpaths(String[] rllibpaths) {
        File[] files = new File[rllibpaths.length];
        for (int i = 0; i < files.length; i++) {
            files[i] = new File(rllibpaths[i]);
        }
        this.rllibpaths = files;
        return this;
    }

    String algorithm() {
        return algorithm;
    }
    RLlibHelper algorithm(String algorithm) {
        this.algorithm = algorithm;
        return this;
    }

    File checkpoint() {
        return checkpoint;
    }
    RLlibHelper checkpoint(File checkpoint) {
        this.checkpoint = checkpoint;
        return this;
    }
    RLlibHelper checkpoint(String checkpoint) {
        this.checkpoint = new File(checkpoint);
        return this;
    }

    Environment environment() {
        return environment;
    }
    RLlibHelper environment(Environment environment) {
        this.environment = environment;
        return this;
    }

    int numGPUs() {
        return numGPUs;
    }
    RLlibHelper numGPUs(int numGPUs) {
        this.numGPUs = numGPUs;
        return this;
    }

    int numWorkers() {
        return numWorkers;
    }
    RLlibHelper numWorkers(int numWorkers) {
        this.numWorkers = numWorkers;
        return this;
    }

    long randomSeed() {
        return randomSeed;
    }
    RLlibHelper randomSeed(long randomSeed) {
        this.randomSeed = randomSeed;
        return this;
    }

    double[] gammas() {
        return gammas;
    }
    RLlibHelper gammas(double[] gammas) {
        this.gammas = gammas;
        return this;
    }

    double[] learningRates() {
        return learningRates;
    }
    RLlibHelper learningRates(double[] learningRates) {
        this.learningRates = learningRates;
        return this;
    }

    int[] miniBatchSizes() {
        return miniBatchSizes;
    }
    RLlibHelper miniBatchSizes(int[] miniBatchSizes) {
        this.miniBatchSizes = miniBatchSizes;
        return this;
    }

    int numHiddenLayers() {
        return numHiddenLayers;
    }
    RLlibHelper numHiddenLayers(int numHiddenLayers) {
        this.numHiddenLayers = numHiddenLayers;
        return this;
    }

    int numHiddenNodes() {
        return numHiddenNodes;
    }
    RLlibHelper numHiddenNodes(int numHiddenNodes) {
        this.numHiddenNodes = numHiddenNodes;
        return this;
    }

    int stepsPerIteration() {
        return stepsPerIteration;
    }
    RLlibHelper stepsPerIteration(int stepsPerIteration) {
        this.stepsPerIteration = stepsPerIteration;
        return this;
    }

    int maxIterations() {
        return maxIterations;
    }
    RLlibHelper maxIterations(int maxIterations) {
        this.maxIterations = maxIterations;
        return this;
    }

    int savePolicyInterval() {
        return savePolicyInterval;
    }
    RLlibHelper savePolicyInterval(int savePolicyInterval) {
        this.savePolicyInterval = savePolicyInterval;
        return this;
    }

    String redisAddress() {
        return redisAddress;
    }
    RLlibHelper redisAddress(String redisAddress) {
        this.redisAddress = redisAddress;
        return this;
    }

    String hiddenLayers() {
        String s = "[";
        for (int i = 0; i < numHiddenLayers; i++) {
            s += numHiddenNodes + (i < numHiddenLayers - 1 ? ", " : "]");
        }
        return s;
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

    public String generatePythonTrainer() {
        if (environment == null) {
            throw new IllegalStateException("Environment is null.");
        }
        String trainer = "import glob, gym, nativerl, numpy, ray, sys, os\n"
            + "from ray.rllib.utils import seed\n"
            + "\n"
            + "jardir = os.getcwd()\n"
            + "\n"
            + "class " + environment.getClass().getSimpleName() + "(gym.Env):\n"
            + "    def __init__(self, env_config):\n"
            + "        # Put all JAR files found here in the class path\n"
            + "        jars = glob.glob(jardir + '/**/*.jar', recursive=True)\n"
            + "        nativerl.init(['-Djava.class.path=' + os.pathsep.join(jars + [jardir])]);\n"
            + "\n"
            + "        self.nativeEnv = nativerl.createEnvironment('" + environment.getClass().getName() + "')\n"
            + "        actionSpace = self.nativeEnv.getActionSpace()\n"
            + "        observationSpace = self.nativeEnv.getObservationSpace()\n"
            + "        self.action_space = gym.spaces.Discrete(actionSpace.n)\n"
            + "        self.observation_space = gym.spaces.Box(observationSpace.low[0], observationSpace.high[0], numpy.array(observationSpace.shape), dtype=numpy.float32)\n"
            + "    def reset(self):\n"
            + "        self.nativeEnv.reset()\n"
            + "        return numpy.array(self.nativeEnv.getObservation())\n"
            + "    def step(self, action):\n"
            + "        reward = self.nativeEnv.step(action)\n"
            + "        return numpy.array(self.nativeEnv.getObservation()), reward, self.nativeEnv.isDone(), {}\n"
            + "\n"
            + "# Make sure multiple processes can read the database from AnyLogic\n"
            + "with open('database/db.properties', 'r+') as f:\n"
            + "    lines = f.readlines()\n"
            + "    if 'hsqldb.lock_file=false\\n' not in lines:\n"
            + "        f.write('hsqldb.lock_file=false\\n')\n"
            + "\n"
            + "ray.init(" + (redisAddress != null ? "redis_address='" + redisAddress + "'" : "") + ")\n"
            + "seed.seed(" + randomSeed + ")\n"
            + "model = ray.rllib.models.MODEL_DEFAULTS.copy()\n"
            + "model['fcnet_hiddens'] = " + hiddenLayers() + "\n"
            + "ray.tune.run(\n"
            + "    '" + algorithm + "',\n"
            + "    stop={'training_iteration': " + maxIterations + "},\n"
            + "    config={\n"
            + "        'env': " + environment.getClass().getSimpleName() + ",\n"
            + "        'num_gpus': " + numGPUs + ",\n"
            + "        'num_workers': " + numWorkers + ",\n"
            + "        'gamma': ray.tune.grid_search(" +  Arrays.toString(gammas) + "),\n"
            + "        'lr': ray.tune.grid_search(" +  Arrays.toString(learningRates) + "),\n"
            + "        'sgd_minibatch_size': ray.tune.grid_search(" + Arrays.toString(miniBatchSizes) + "),\n"
            + "        'model': model,\n"
            + "        'train_batch_size': " + stepsPerIteration + ",\n"
            + "    },\n"
            + (checkpoint != null ? "    restore='" + checkpoint.getAbsolutePath() + "')\n" : "")
            + "    checkpoint_freq=" + savePolicyInterval + ",\n"
            + "    export_formats=['model'], # Export TensorFlow SavedModel as well\n"
            + ")\n";
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
                System.out.println("    --random-seed");
                System.out.println("    --gamma");
                System.out.println("    --learning-rate");
                System.out.println("    --mini-batch-size");
                System.out.println("    --num-hidden-layers");
                System.out.println("    --num-hidden-nodes");
                System.out.println("    --steps-per-iteration");
                System.out.println("    --max-iterations");
                System.out.println("    --save-policy-interval");
                System.out.println("    --redis-address");
                System.exit(0);
            } else if ("--rllibpaths".equals(args[i])) {
                helper.rllibpaths(args[++i].split(File.pathSeparator));
            } else if ("--algorithm".equals(args[i])) {
                helper.algorithm(args[++i]);
            } else if ("--checkpoint".equals(args[i])) {
                helper.checkpoint(args[++i]);
            } else if ("--environment".equals(args[i])) {
                helper.environment(Class.forName(args[++i]).asSubclass(Environment.class).newInstance());
            } else if ("--num-gpus".equals(args[i])) {
                helper.numGPUs(Integer.parseInt(args[++i]));
            } else if ("--num-workers".equals(args[i])) {
                helper.numWorkers(Integer.parseInt(args[++i]));
            } else if ("--random-seed".equals(args[i])) {
                helper.randomSeed(Long.parseLong(args[++i]));
            } else if ("--gamma".equals(args[i])) {
                String[] strings = args[++i].split(",");
                double[] doubles = new double[strings.length];
                for (int j = 0; j < doubles.length; j++) {
                    doubles[j] = Double.parseDouble(strings[j]);
                }
                helper.gammas(doubles);
            } else if ("--learning-rate".equals(args[i])) {
                String[] strings = args[++i].split(",");
                double[] doubles = new double[strings.length];
                for (int j = 0; j < doubles.length; j++) {
                    doubles[j] = Double.parseDouble(strings[j]);
                }
                helper.learningRates(doubles);
            } else if ("--mini-batch-size".equals(args[i])) {
                String[] strings = args[++i].split(",");
                int[] ints = new int[strings.length];
                for (int j = 0; j < ints.length; j++) {
                    ints[j] = Integer.parseInt(strings[j]);
                }
                helper.miniBatchSizes(ints);
            } else if ("--num-hidden-layers".equals(args[i])) {
                helper.numHiddenLayers(Integer.parseInt(args[++i]));
            } else if ("--num-hidden-nodes".equals(args[i])) {
                helper.numHiddenNodes(Integer.parseInt(args[++i]));
            } else if ("--steps-per-iteration".equals(args[i])) {
                helper.stepsPerIteration(Integer.parseInt(args[++i]));
            } else if ("--max-iterations".equals(args[i])) {
                helper.maxIterations(Integer.parseInt(args[++i]));
            } else if ("--save-policy-interval".equals(args[i])) {
                helper.savePolicyInterval(Integer.parseInt(args[++i]));
            } else if ("--redis-address".equals(args[i])) {
                helper.redisAddress(args[++i]);
            } else {
                output = new File(args[i]);
            }
        }
        helper.generatePythonTrainer(output);
    }
}
