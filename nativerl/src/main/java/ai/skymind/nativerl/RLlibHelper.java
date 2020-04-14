package ai.skymind.nativerl;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.bytedeco.cpython.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;
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
    long randomSeed = 0;
    int numHiddenLayers = 2;
    int numHiddenNodes = 256;
    int maxIterations = 500;
    int maxTimeInSec = -1;
    int numSamples = 10;
    double maxRewardMean = Double.POSITIVE_INFINITY;
    int savePolicyInterval = 100;
    String redisAddress = null;
    String customParameters = "";
    boolean resume = false;
    int checkpointFrequency = 50;
    boolean multiAgent = false;
    boolean userLog = false;

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
        this.randomSeed = copy.randomSeed;
        this.numHiddenLayers = copy.numHiddenLayers;
        this.numHiddenNodes = copy.numHiddenNodes;
        this.maxIterations = copy.maxIterations;
        this.maxRewardMean = copy.maxRewardMean;
        this.savePolicyInterval = copy.savePolicyInterval;
        this.redisAddress = copy.redisAddress;
        this.maxTimeInSec = copy.maxTimeInSec;
        this.customParameters = copy.customParameters;
        this.numSamples = copy.numSamples;
        this.resume = copy.resume;
        this.checkpointFrequency = copy.checkpointFrequency;
        this.userLog = userLog;
    }

    @Override public String toString() {
        return "RLlibHelper[rllibpaths=" + Arrays.deepToString(rllibpaths) + ", "
                + "algorithm=" + algorithm + ", "
                + "outputDir=" + outputDir + ", "
                + "checkpoint=" + checkpoint + ", "
                + "environment=" + environment + ", "
                + "numGPUs=" + numGPUs + ", "
                + "numWorkers=" + numWorkers + ", "
                + "randomSeed=" + randomSeed + ", "
                + "numHiddenLayers=" + numHiddenLayers + ", "
                + "numHiddenNodes=" + numHiddenNodes  + ", "
                + "maxIterations=" + maxIterations + ", "
                + "maxRewardMean=" + maxRewardMean + ", "
                + "savePolicyInterval=" + savePolicyInterval + ", "
                + "maxTimeInSec=" + maxTimeInSec + ", "
                + "redisAddress=" + redisAddress + ", "
                + "resume=" + resume + ", "
                + "checkpointFrequency=" + checkpointFrequency + ", "
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

    public long randomSeed() {
        return randomSeed;
    }
    public RLlibHelper randomSeed(long randomSeed) {
        this.randomSeed = randomSeed;
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

    public double maxRewardMean() {
        return maxRewardMean;
    }
    public RLlibHelper maxRewardMean(double maxRewardMean) {
        this.maxRewardMean = maxRewardMean;
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
        String trainer = "import glob, gym, nativerl, ray, sys, os, random\n"
            + "import numpy as np\n"
            + "from ray.rllib.env import MultiAgentEnv\n"
            + "from ray.rllib.agents.registry import get_agent_class\n"
            + "from ray.rllib.utils import seed\n"
            + "from ray.tune import run, sample_from\n"
            + "from ray.tune.schedulers import PopulationBasedTraining\n"
            + "\n"
            + "jardir = os.getcwd()\n"
            + "\n"
            + "class " + environment.getClass().getSimpleName() + "(" + (multiAgent ? "MultiAgentEnv" : "gym.Env") + "):\n"
            + "    def __init__(self, env_config):\n"
            + "        # Put all JAR files found here in the class path\n"
            + "        jars = glob.glob(jardir + '/**/*.jar', recursive=True)\n"
            + "        nativerl.init(['-Djava.class.path=' + os.pathsep.join(jars + [jardir])]);\n"
            + "\n"
            + "        self.nativeEnv = nativerl.createEnvironment('" + environment.getClass().getName() + "')\n"
            + "        actionSpace = self.nativeEnv.getActionSpace()\n"
            + "        observationSpace = self.nativeEnv.getObservationSpace()\n"
            + "        self.action_space = gym.spaces.Discrete(actionSpace.n)\n"
            + "        self.observation_space = gym.spaces.Box(observationSpace.low[0], observationSpace.high[0], np.array(observationSpace.shape), dtype=np.float32)\n"
            + "        self.id = '" + environment.getClass().getSimpleName() + "'\n"
            + "        self.max_episode_steps = " + Integer.MAX_VALUE + "\n"
            + (multiAgent ? "" : "        self.unwrapped.spec = self\n")
            + "    def reset(self):\n"
            + "        self.nativeEnv.reset()\n"
            + (multiAgent
                ? "        obs = np.array(self.nativeEnv.getObservation())\n"
                + "        obsdict = {}\n"
                + "        for i in range(0, obs.shape[0]):\n"
                + "            obsdict[str(i)] = obs[i]\n"
                + "        return obsdict\n"

                : "        return np.array(self.nativeEnv.getObservation())\n")
            + "    def step(self, action):\n"
            + (multiAgent
                ? "        actionarray = np.ndarray(shape=(len(action), 1), dtype=np.float32)\n"
                + "        for i in range(0, actionarray.shape[0]):\n"
                + "            actionarray[i,:] = action[str(i)].astype(np.float32)\n"
                + "        reward = np.array(self.nativeEnv.step(nativerl.Array(actionarray)))\n"
                + "        obs = np.array(self.nativeEnv.getObservation())\n"
                + "        obsdict = {}\n"
                + "        rewarddict = {}\n"
                + "        for i in range(0, obs.shape[0]):\n"
                + "            obsdict[str(i)] = obs[i]\n"
                + "            rewarddict[str(i)] = reward[i]\n"
                + "        return obsdict, rewarddict, {'__all__' : self.nativeEnv.isDone()}, {}\n"

                : "        reward = self.nativeEnv.step(action)\n"
                + "        return np.array(self.nativeEnv.getObservation()), reward, self.nativeEnv.isDone(), {}\n")
                + "\n"
                + "class Stopper:\n"
                + "    def __init__(self):\n"
                + "        # Core criteria\n"
                + "        self.should_stop = False # Stop criteria met\n"
                + "        self.too_many_iter = False # Max iterations\n"
                + "        self.too_much_time = False # Max training time\n"
                + "        self.too_many_steps = False # Max steps\n"
                + "\n"
                + "        # Stopping criteria at self.early_check\n"
                + "        self.no_discovery_risk = False # Value loss never changes\n"
                + "        self.no_converge_risk = False # Entropy never drops\n"
                + "\n"
                + "        # Convergence signals at each iteration from self.converge_check onward\n"
                + "        self.converged = False # Reward mean changes very little\n"
                + "        #self.value_pred_adequate = False # Explained variance >= self.value_pred_threshold\n"
                + "\n"
                + "        # Episode reward behaviour\n"
                + "        self.episode_reward_window = []\n"
                + "        self.episode_reward_range = 0\n"
                + "        self.episode_reward_mean = 0\n"
                + "        self.episode_reward_mean_latest = 0\n"
                + "\n"
                + "        # Entropy behaviour\n"
                + "        self.entropy_start = 0\n"
                + "        self.entropy_now = 0\n"
                + "        self.entropy_slope = 0\n"
                + "\n"
                + "        # Value loss behaviour\n"
                + "        self.vf_loss_window = []\n"
                + "        self.vf_loss_range = 0\n"
                + "\n"
                + "        # Configs\n"
                + "        self.episode_reward_range_threshold = 0.1 # Remove with 0\n"
                + "        self.entropy_slope_threshold = 0.01 # Remove with -9999999\n"
                + "        self.vf_loss_range_threshold = 0.1 # Remove with 0\n"
                + "        #self.value_pred_threshold = 0.5 # Remove with -9999999\n"
                + "\n"
                + "    def stop(self, trial_id, result):\n"
                + "        # Core Criteria\n"
                + "        self.too_many_iter = result['training_iteration'] >= " + maxIterations + "\n"
                + (maxTimeInSec > 0
                ? "        self.too_much_time = result['time_total_s'] >= " + maxTimeInSec + "\n"
                : "")
                + "\n"
                + "        if not self.should_stop and (self.too_many_iter or self.too_much_time):\n"
                + "            self.should_stop = True\n"
                + "            return self.should_stop\n"
                + "\n"
                + "        # Append episode rewards list used for trend behaviour\n"
                + "        self.episode_reward_window.append(result['episode_reward_mean'])\n"
                + "\n"
                + "        # Up until early stopping filter, append value loss list to measure range\n"
                + "        if result['training_iteration'] <= 50:\n"
                + "            self.vf_loss_window.append(result['info/learner/default_policy/vf_loss'])\n"
                + "\n"
                + "        # Experimental Criteria\n"
                + "\n"
                + "        # Episode steps filter\n"
                + "        if result['training_iteration'] == 1:\n"
                + "            self.entropy_start = result['info/learner/default_policy/entropy'] # Set start value\n"
                + "            # Too many steps within episode\n"
                + "            self.too_many_steps = result['timesteps_this_iter'] > 200000 # Max steps\n"
                + "            if not self.should_stop and self.too_many_steps:\n"
                + "                self.should_stop = True\n"
                + "                return self.should_stop\n"
                + "\n"
                + "        # Early stopping filter\n"
                + "        if result['training_iteration'] == 50:\n"
                + "            self.entropy_now = result['info/learner/default_policy/entropy']\n"
                + "            self.episode_reward_range = np.max(np.array(self.episode_reward_window)) - np.min(np.array(self.episode_reward_window))\n"
                + "            self.entropy_slope = self.entropy_now - self.entropy_start\n"
                + "            self.vf_loss_range = np.max(np.array(self.vf_loss_window)) - np.min(np.array(self.vf_loss_window))\n"
                + "            if np.abs(self.episode_reward_range) < np.abs(self.episode_reward_window[0] * self.episode_reward_range_threshold):\n"
                + "                self.no_converge_risk = True\n"
                + "            elif self.entropy_slope > np.abs(self.entropy_start * self.entropy_slope_threshold):\n"
                + "                self.no_converge_risk = True\n"
                + "            if np.abs(self.vf_loss_range) < np.abs(self.vf_loss_window[0] * self.vf_loss_range_threshold):\n"
                + "                self.no_discovery_risk = True\n"
                + "\n"
                + "            # Early stopping decision\n"
                + "            if not self.should_stop and (self.no_converge_risk or self.no_discovery_risk):\n"
                + "                self.should_stop = True\n"
                + "                return self.should_stop\n"
                + "\n"
                + "        # Convergence Filter\n"
                + "        if result['training_iteration'] >= 100:\n"
                + "            # Episode reward range activity\n"
                + "            self.episode_reward_range = np.max(np.array(self.episode_reward_window[-50:])) - np.min(np.array(self.episode_reward_window[-50:]))\n"
                + "            # Episode reward mean activity\n"
                + "            self.episode_reward_mean = np.mean(np.array(self.episode_reward_window[-50:]))\n"
                + "            self.episode_reward_mean_latest = np.mean(np.array(self.episode_reward_window[-5:]))\n"
                + "\n"
                + "            # Convergence check\n"
                + "            if (np.abs(self.episode_reward_mean_latest - self.episode_reward_mean) / np.abs(self.episode_reward_mean) < self.episode_reward_range_threshold) and (np.abs(self.episode_reward_range) < np.abs(np.mean(np.array(self.episode_reward_window[-50:])) * 2)):\n"
                + "                self.converged = True\n"
                + "\n"
                + "            # Optional: Extra convergence check\n"
                + "            #self.value_pred_adequate = (np.abs(result['info/learner/default_policy/vf_explained_var']) > self.value_pred_threshold)\n"
                + "\n"
                + "            # Convergence stopping decision\n"
                + "            if not self.should_stop and self.converged: #and self.value_pred_adequate\n"
                + "                self.should_stop = True\n"
                + "                return self.should_stop\n"
                + "\n"
                + "        # Returns False by default until stopping decision made\n"
                + "        return self.should_stop\n"
                + "\n"
                + "stopper = Stopper()\n"
                + "\n"
                + "pbt_scheduler = PopulationBasedTraining(\n"
                + "    time_attr = 'training_iteration',\n"
                + "    metric = 'episode_reward_mean',\n"
                + "    mode = 'max',\n"
                + "    perturbation_interval = 10,\n"
                + "    quantile_fraction = 0.25,\n"
                + "    resample_probability = 0.25,\n"
                + "    log_config = True,\n"
                + "    hyperparam_mutations = {\n"
                + "        'lambda': np.linspace(0.9, 1.0, 5).tolist(),\n"
                + "        'clip_param': np.linspace(0.01, 0.5, 5).tolist(),\n"
                + "        'entropy_coeff': np.linspace(0, 0.03, 5).tolist(),\n"
                + "        'lr': [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],\n"
                + "        'num_sgd_iter': [5, 10, 15, 20, 30],\n"
                + "        'sgd_minibatch_size': [128, 256, 512, 1024, 2048],\n"
                + "        'train_batch_size': [4000, 6000, 8000, 10000, 12000]\n"
                + "    }\n"
                + ")\n"
            + "\n"
            + "# Make sure multiple processes can read the database from AnyLogic\n"
            + "with open('database/db.properties', 'r+') as f:\n"
            + "    lines = f.readlines()\n"
            + "    if 'hsqldb.lock_file=false\\n' not in lines:\n"
            + "        f.write('hsqldb.lock_file=false\\n')\n"
            + "\n"
            + "ray.init(log_to_driver=" + (userLog ? "True" : "False") + ")\n"
            + "seed.seed(" + randomSeed + ")\n"
            + "model = ray.rllib.models.MODEL_DEFAULTS.copy()\n"
            + "model['fcnet_hiddens'] = " + hiddenLayers() + "\n"
            + "\n"
            + "trials = run(\n"
            + "    'PPO',\n"
            + "    scheduler = pbt_scheduler,\n"
            + "    num_samples = " + numSamples + ",\n"
            + "    stop = stopper.stop,\n"
            + "    config = {\n"
            + "        'env': " + environment.getClass().getSimpleName() + ",\n"
            + "        'num_gpus': 0,\n"
            + "        'num_workers': 1,\n"
            + "        'model': model,\n"
            + "        'use_gae': True,\n"
            + "        'vf_loss_coeff': 1.0,\n"
            + "        'vf_clip_param': np.inf,\n"
            + "        # These params are tuned from a fixed starting value.\n"
            + "        'lambda': 0.95,\n"
            + "        'clip_param': 0.2,\n"
            + "        'lr': 1e-4,\n"
            + "        'entropy_coeff': 0.0,\n"
            + "        # These params start off randomly drawn from a set.\n"
            + "        'num_sgd_iter': sample_from(\n"
            + "                lambda spec: random.choice([10, 20, 30])),\n"
            + "        'sgd_minibatch_size': sample_from(\n"
            + "                lambda spec: random.choice([128, 512, 2048])),\n"
            + "        'train_batch_size': sample_from(\n"
            + "                lambda spec: random.choice([4000, 8000, 12000])),\n"
            + "        # Set rollout samples to episode length\n"
            + "        'batch_mode': 'complete_episodes',\n"
            + "        # Auto-normalize observations\n"
            + "        #'observation_filter': 'MeanStdFilter'\n"
            + "    },\n"
            + (outputDir != null ? "    local_dir = '" + outputDir.getAbsolutePath() + "',\n" : "")
            + "    resume = " + (resume ? "True" : "False") + ",\n"
            + "    checkpoint_freq = " + checkpointFrequency + ",\n"
            + "    checkpoint_at_end = True,\n"
            + "    max_failures = 1,\n"
            + "    export_formats = ['model']\n"
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
                System.out.println("    --num-hidden-layers");
                System.out.println("    --num-hidden-nodes");
                System.out.println("    --max-iterations");
                System.out.println("    --max-reward-mean");
                System.out.println("    --save-policy-interval");
                System.out.println("    --redis-address");
                System.out.println("    --custom-parameters");
                System.out.println("    --multi-agent");
                System.out.println("    --maxTimeInSec");
                System.out.println("    --num-samples");
                System.out.println("    --resume");
                System.out.println("    --checkpoint-freqeuncy");
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
            } else if ("--random-seed".equals(args[i])) {
                helper.randomSeed(Long.parseLong(args[++i]));
            } else if ("--num-hidden-layers".equals(args[i])) {
                helper.numHiddenLayers(Integer.parseInt(args[++i]));
            } else if ("--num-hidden-nodes".equals(args[i])) {
                helper.numHiddenNodes(Integer.parseInt(args[++i]));
            } else if ("--max-iterations".equals(args[i])) {
                helper.maxIterations(Integer.parseInt(args[++i]));
            } else if ("--max-reward-mean".equals(args[i])) {
                helper.maxRewardMean(Double.parseDouble(args[++i]));
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
            } else if ("--user-log".equals(args[i])) {
                helper.userLog = true;
            } else {
                output = new File(args[i]);
            }
        }
        helper.generatePythonTrainer(output);
    }
}
