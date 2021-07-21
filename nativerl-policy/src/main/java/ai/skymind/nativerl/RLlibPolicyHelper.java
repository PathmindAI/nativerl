package ai.skymind.nativerl;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.tensorflow.*;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.bytedeco.tensorflow.global.tensorflow.*;

/**
 * A PolicyHelper for RLlib, which can load only TensorFlow SavedModel exported by RLlib.
 * Does not require CPython so has none of its limitations, such as the GIL.
 */
public class RLlibPolicyHelper implements PolicyHelper {
    final String[] inputNames = {"observations", "prev_action", "prev_reward", "is_training", "seq_lens"};
    final String[] outputNames = {"actions", "action_prob", "action_dist_inputs", "vf_preds", "action_logp"};

    SavedModelBundle bundle = null;
    SessionOptions options = null;
    RunOptions runOptions = null;
    StringUnorderedSet tags = null;
    int[] inputTensorTypes = null;
    TensorShape[] inputTensorShapes = null;
    String[] realInputNames = null;
    String[] realOutputNames = null;
    ThreadLocal<Tensor[]> inputTensors = new ThreadLocal<Tensor[]>() {
        @Override protected Tensor[] initialValue() {
            Tensor[] tensors = new Tensor[inputNames.length];
            for (int i = 0; i < tensors.length; i++) {
                tensors[i] = new Tensor(inputTensorTypes[i], inputTensorShapes[i]);
            }
            return tensors;
        }
    };
    ThreadLocal<TensorVector> outputTensors = new ThreadLocal<TensorVector>() {
        @Override protected TensorVector initialValue() {
            return new TensorVector(realOutputNames.length);
        }
    };
    int actionTupleSize;

    public RLlibPolicyHelper(File savedModel) throws IOException {
        if (disablePolicyHelper) {
            return;
        }
        bundle = new SavedModelBundle();
        options = new SessionOptions();
        runOptions = new RunOptions();
        tags = new StringUnorderedSet();
        tags.insert(kSavedModelTagServe());
        Status s = LoadSavedModel(options, runOptions, savedModel.getAbsolutePath(), tags, bundle);
        if (!s.ok()) {
            throw new IOException(s.error_message().getString());
        }

        StringSignatureDefMap map = bundle.meta_graph_def().signature_def();
        SignatureDef signature = map.get(kDefaultServingSignatureDefKey());
//        System.out.println("inputs:");
//        for (StringTensorInfoMap.Iterator it = signature.inputs().begin(); !it.equals(signature.inputs().end()); it.increment()) {
//            System.out.println(it.first().getString());
//        }
//        System.out.println("outputs:");
//        for (StringTensorInfoMap.Iterator it = signature.outputs().begin(); !it.equals(signature.outputs().end()); it.increment()) {
//            System.out.println(it.first().getString());
//        }

        realInputNames = new String[inputNames.length];
        inputTensorTypes = new int[inputNames.length];
        inputTensorShapes = new TensorShape[inputNames.length];
        for (int i = 0; i < inputNames.length; i++) {
            TensorInfo info = signature.inputs().get(new BytePointer(inputNames[i]));
            realInputNames[i] = info.name().getString();
            inputTensorTypes[i] = info.dtype();
            TensorShapeProto tsp = info.mutable_tensor_shape();
            if (tsp.dim_size() > 0) {
                TensorShapeProto_Dim dim = tsp.mutable_dim(0);
                if (dim.size() < 0) {
                    // use minibatch sizes of 1
                    dim.set_size(1);
                }
            }
            inputTensorShapes[i] = new TensorShape(tsp);
        }

        // initialize output names along with action tuple size
        List<String> tempOutputNames = new ArrayList<>();

        // we may need this logic for Ray 0.9.0
        TensorInfo info = signature.outputs().get(new BytePointer(outputNames[0]));
        if (info != null && info.name() != null) {
            actionTupleSize = 1;
            tempOutputNames.add(info.name().getString());
        } else {
            actionTupleSize = 0;
            while ((info = signature.outputs().get(new BytePointer(outputNames[0] + "_" + actionTupleSize))) != null
                    && info.name() != null) {
                actionTupleSize++;
                tempOutputNames.add(info.name().getString());
            }
        }
        for (int i = 1; i < outputNames.length; i++) {
            tempOutputNames.add(signature.outputs().get(new BytePointer(outputNames[i])).name().getString());
        }
        realOutputNames = tempOutputNames.toArray(new String[tempOutputNames.size()]);
    }

    @Override public float[] computeActions(float[] state) {
        if (disablePolicyHelper) {
            return null;
        }
        if (state.length != inputTensorShapes[0].num_elements()) {
            throw new IllegalArgumentException("Array length not equal to model input size: " + state.length + " != " + inputTensorShapes[0].num_elements());
        }
        new FloatPointer(inputTensors.get()[0].tensor_data()).put(state);

        Status s = bundle.session().Run(new StringTensorPairVector(realInputNames, inputTensors.get()),
                new StringVector(realOutputNames), new StringVector(), outputTensors.get());
        if (!s.ok()) {
            throw new RuntimeException(s.error_message().getString());
        }

        long numActionElements = 0;
        Tensor[] actions = new Tensor[actionTupleSize];
        for (int i = 0; i < actionTupleSize; i++) {
            Tensor t = outputTensors.get().get(i);
            numActionElements += t.NumElements();
            actions[i] = t;
        }

        int k = 0;
        float[] actionArray = new float[(int)numActionElements];
        for (int i = 0; i < actionTupleSize; i++) {
            Tensor t = actions[i];
            switch (t.dtype()) {
                case DT_INT32:
                case DT_UINT32:
                    IntPointer ip = new IntPointer(t.tensor_data());
                    for (int j = 0; j < t.NumElements(); j++) {
                        actionArray[k++] = ip.get(j);
                    }
                    break;
                case DT_INT64:
                case DT_UINT64:
                    LongPointer lp = new LongPointer(t.tensor_data());
                    for (int j = 0; j < t.NumElements(); j++) {
                        actionArray[k++] = lp.get(j);
                    }
                    break;
                case DT_FLOAT:
                    FloatPointer fp = new FloatPointer(t.tensor_data());
                    for (int j = 0; j < t.NumElements(); j++) {
                        actionArray[k++] = Math.max(0.0f, Math.min(1.0f, fp.get(j)));
                    }
                    break;
                case DT_DOUBLE:
                    DoublePointer dp = new DoublePointer(t.tensor_data());
                    for (int j = 0; j < t.NumElements(); j++) {
                        actionArray[k++] = Math.max(0.0f, Math.min(1.0f, (float)dp.get(j)));
                    }
                    break;
                default:
                    throw new UnsupportedOperationException("Tensor.dtype() == " + t.dtype());
            }
        }
//        for (int i = 0; i < outputTensors.size(); i++) {
//            System.out.println(outputTensors.get().get(i).createIndexer());
//        }
        return actionArray;
    }

    @Override public long[] computeDiscreteAction(float[] state) {
        if (disablePolicyHelper) {
            return null;
        }
        if (state.length != inputTensorShapes[0].num_elements()) {
            throw new IllegalArgumentException("Array length not equal to model input size: " + state.length + " != " + inputTensorShapes[0].num_elements());
        }
        new FloatPointer(inputTensors.get()[0].tensor_data()).put(state);

        long[] actionArray = new long[actionTupleSize];
        Status s = bundle.session().Run(new StringTensorPairVector(realInputNames, inputTensors.get()),
                new StringVector(realOutputNames), new StringVector(), outputTensors.get());
        if (!s.ok()) {
            throw new RuntimeException(s.error_message().getString());
        }
        for (int i = 0; i < actionArray.length; i++) {
            actionArray[i] = new LongPointer(outputTensors.get().get(i).tensor_data()).get();
        }
//        for (int i = 0; i < outputTensors.size(); i++) {
//            System.out.println(outputTensors.get().get(i).createIndexer());
//        }
        return actionArray;
    }

    @Override
    public double[] computeActions(String url, String token, String postBody) {
        throw new UnsupportedOperationException("Unsupported method for RLlibV2PolicyHelper");
    }

}
