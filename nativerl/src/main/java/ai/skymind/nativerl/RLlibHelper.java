package ai.skymind.nativerl;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import org.bytedeco.cpython.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;
import org.bytedeco.numpy.*;
import static org.bytedeco.cpython.global.python.*;
import static org.bytedeco.numpy.global.numpy.*;

public class RLlibHelper implements PolicyHelper {
    PyObject globals = null;
    PyArrayObject obsArray = null;
    FloatPointer obsData = null;

    public RLlibHelper(File[] rllibpaths, String algorithm, File checkpoint, Environment env) throws IOException {
        this(rllibpaths, algorithm, checkpoint, env.getClass().getSimpleName(), env.getActionSpace(), env.getObservationSpace());
    }
    public RLlibHelper(File[] rllibpaths, String algorithm, File checkpoint, String name, long discreteActions, long continuousObservations) throws IOException {
        this(rllibpaths, algorithm, checkpoint, name, AbstractEnvironment.getDiscreteSpace(discreteActions),
                AbstractEnvironment.getContinuousSpace(continuousObservations));
    }
    public RLlibHelper(File[] rllibpaths, String algorithm, File checkpoint, String name, Space actionSpace, Space obsSpace) throws IOException {
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
