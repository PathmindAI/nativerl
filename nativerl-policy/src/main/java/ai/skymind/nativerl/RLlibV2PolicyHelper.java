package ai.skymind.nativerl;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.NdArray;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.buffer.DataBuffers;
import org.tensorflow.ndarray.buffer.DoubleDataBuffer;
import org.tensorflow.ndarray.buffer.FloatDataBuffer;
import org.tensorflow.ndarray.buffer.IntDataBuffer;
import org.tensorflow.ndarray.buffer.LongDataBuffer;
import org.tensorflow.proto.framework.DataType;
import org.tensorflow.proto.framework.SignatureDef;
import org.tensorflow.proto.framework.TensorInfo;
import org.tensorflow.proto.framework.TensorShapeProto;
import org.tensorflow.types.TBool;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;

/**
 * A PolicyHelper for RLlib, which can load only TensorFlow 2.x SavedModel exported by RLlib.
 * Does not require CPython so has none of its limitations, such as the GIL.
 */
public class RLlibV2PolicyHelper implements PolicyHelper {
    final String[] inputNames = {"observations", "prev_action", "prev_reward", "is_training", "seq_lens"};
    final String[] outputNames = {"actions", "action_prob", "action_dist_inputs", "vf_preds", "action_logp"};

    SavedModelBundle bundle = null;
    int[] inputTensorTypes = null;
    Shape[] inputTensorShapes = null;
    String[] realInputNames = null;
    String[] realOutputNames = null;
    ThreadLocal<Tensor[]> inputTensors = new ThreadLocal<Tensor[]>() {
        @Override protected Tensor[] initialValue() {
            Tensor[] tensors = new Tensor[inputNames.length];
            for (int i = 0; i < tensors.length; i++) {
                org.tensorflow.DataType dt;
                switch (inputTensorTypes[i]) {
                    case DataType.DT_BOOL_VALUE:   dt = TBool.DTYPE; break;
                    case DataType.DT_DOUBLE_VALUE: dt = TFloat64.DTYPE; break;
                    case DataType.DT_FLOAT_VALUE:  dt = TFloat32.DTYPE; break;
                    case DataType.DT_INT32_VALUE:  dt = TInt32.DTYPE; break;
                    case DataType.DT_INT64_VALUE:  dt = TInt64.DTYPE; break;
                    default: throw new UnsupportedOperationException("Unsupported input data type: " + inputTensorTypes[i]);
                }
                tensors[i] = Tensor.of(dt, inputTensorShapes[i]);
            }
            return tensors;
        }
    };
    int actionTupleSize;

    public RLlibV2PolicyHelper(File savedModel) throws IOException {
        if (disablePolicyHelper) {
            return;
        }
        bundle = SavedModelBundle.load(savedModel.getAbsolutePath(), "serve");
        SignatureDef signature = bundle.metaGraphDef().getSignatureDefOrThrow("serving_default");
//        System.out.println(signature);

        realInputNames = new String[inputNames.length];
        inputTensorTypes = new int[inputNames.length];
        inputTensorShapes = new Shape[inputNames.length];
        for (int i = 0; i < inputNames.length; i++) {
            TensorInfo info = signature.getInputsMap().get(inputNames[i]);
            realInputNames[i] = info.getName();
            inputTensorTypes[i] = info.getDtypeValue();
            TensorShapeProto tsp = info.getTensorShape();
            if (tsp.getDimCount() > 0) {
                long[] dims = new long[tsp.getDimCount()];
                for (int j = 0; j < dims.length; j++) {
                    dims[j] = tsp.getDim(j).getSize();
                    if (dims[j] < 0) {
                        // use minibatch sizes of 1
                        dims[j] = 1;
                    }
                }
                inputTensorShapes[i] = Shape.of(dims);
            } else {
                inputTensorShapes[i] = Shape.scalar();
            }
        }

        // initialize output names along with action tuple size
        List<String> tempOutputNames = new ArrayList<>();

        // we may need this logic for Ray 0.9.0
        TensorInfo info = signature.getOutputsMap().get(outputNames[0]);
        if (info != null && info.getName() != null && info.getName().length() > 0) {
            actionTupleSize = 1;
            tempOutputNames.add(info.getName());
        } else {
            actionTupleSize = 0;
            while ((info = signature.getOutputsMap().get(outputNames[0] + "_" + actionTupleSize)) != null
                    && info.getName() != null && info.getName().length() > 0) {
                actionTupleSize++;
                tempOutputNames.add(info.getName());
            }
        }
        for (int i = 1; i < outputNames.length; i++) {
            tempOutputNames.add(signature.getOutputsMap().get(outputNames[i]).getName());
        }
        realOutputNames = tempOutputNames.toArray(new String[tempOutputNames.size()]);
    }

    @Override public float[] computeActions(float[] state) {
        if (disablePolicyHelper) {
            return null;
        }
        if (state.length != inputTensorShapes[0].size()) {
            throw new IllegalArgumentException("Array length not equal to model input size: " + state.length + " != " + inputTensorShapes[0].size());
        }
        ((TFloat32)inputTensors.get()[0].data()).write(DataBuffers.of(state));

        Session.Runner runner = bundle.session().runner();
        for (int i = 0; i < realInputNames.length; i++) {
            runner.feed(realInputNames[i], inputTensors.get()[i]);
        }
        for (int i = 0; i < realOutputNames.length; i++) {
            runner.fetch(realOutputNames[i]);
        }
        List<Tensor<?>> outputs = runner.run();

        long numActionElements = 0;
        Tensor[] actions = new Tensor[actionTupleSize];
        for (int i = 0; i < actionTupleSize; i++) {
            Tensor t = outputs.get(i);
            numActionElements += ((NdArray)t.data()).size();
            actions[i] = t;
        }

        int k = 0;
        float[] actionArray = new float[(int)numActionElements];
        for (int i = 0; i < actionTupleSize; i++) {
            Tensor t = actions[i];
            if (t.dataType().equals(TFloat64.DTYPE)) {
                DoubleDataBuffer doubles = outputs.get(i).rawData().asDoubles();
                for (int j = 0; j < doubles.size(); j++) {
                    actionArray[k++] = Math.max(0.0f, Math.min(1.0f, (float)doubles.getDouble(j)));
                }
            } else if (t.dataType().equals(TFloat32.DTYPE)) {
                FloatDataBuffer floats = outputs.get(i).rawData().asFloats();
                for (int j = 0; j < floats.size(); j++) {
                    actionArray[k++] = Math.max(0.0f, Math.min(1.0f, floats.getFloat(j)));
                }
            } else if (t.dataType().equals(TInt32.DTYPE)) {
                IntDataBuffer ints = outputs.get(i).rawData().asInts();
                for (int j = 0; j < ints.size(); j++) {
                    actionArray[k++] = ints.getInt(j);
                }
            } else if (t.dataType().equals(TInt64.DTYPE)) {
                LongDataBuffer longs = outputs.get(i).rawData().asLongs();
                for (int j = 0; j < longs.size(); j++) {
                    actionArray[k++] = longs.getLong(j);
                }
            } else {
                throw new UnsupportedOperationException("Unsupported output data type: " + t.dataType());
            }
        }
//        for (int i = 0; i < outputs.size(); i++) {
//            System.out.println(outputTensors.get(i).rawData().asLongs());
//        }
        return actionArray;
    }

    @Override public long[] computeDiscreteAction(float[] state) {
        if (disablePolicyHelper) {
            return null;
        }
        if (state.length != inputTensorShapes[0].size()) {
            throw new IllegalArgumentException("Array length not equal to model input size: " + state.length + " != " + inputTensorShapes[0].size());
        }
        ((TFloat32)inputTensors.get()[0].data()).write(DataBuffers.of(state));

        long[] actionArray = new long[actionTupleSize];
        Session.Runner runner = bundle.session().runner();
        for (int i = 0; i < realInputNames.length; i++) {
            runner.feed(realInputNames[i], inputTensors.get()[i]);
        }
        for (int i = 0; i < realOutputNames.length; i++) {
            runner.fetch(realOutputNames[i]);
        }
        List<Tensor<?>> outputs = runner.run();

        for (int i = 0; i < actionArray.length; i++) {
            actionArray[i] = outputs.get(i).rawData().asLongs().getLong(0);
        }
//        for (int i = 0; i < outputs.size(); i++) {
//            System.out.println(outputTensors.get(i).rawData().asLongs());
//        }
        return actionArray;
    }

}
