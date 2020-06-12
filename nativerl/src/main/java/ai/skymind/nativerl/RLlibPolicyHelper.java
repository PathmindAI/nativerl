package ai.skymind.nativerl;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.tensorflow.*;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * A PolicyHelper for RLlib, which can load only TensorFlow SavedModel exported by RLlib.
 * Does not require CPython so has none of its limitations, such as the GIL.
 */
import static org.bytedeco.tensorflow.global.tensorflow.*;

public class RLlibPolicyHelper implements PolicyHelper {
    final String[] inputNames = {"observations", "prev_action", "prev_reward", "is_training", "seq_lens"};
    final String[] outputNames;

    SavedModelBundle bundle = null;
    SessionOptions options = null;
    RunOptions runOptions = null;
    StringUnorderedSet tags = null;
    TensorShape[] inputTensorShapes = null;
    String[] realInputNames = null;
    String[] realOutputNames = null;
    Tensor[] inputTensors = null;
    FloatPointer obsData = null;
    TensorVector[] outputTensors = null;
    int actionTupleSize;

    public RLlibPolicyHelper(File savedModel) throws IOException {
        this(savedModel, 1);
    }

    public RLlibPolicyHelper(File savedModel, int actionTupleSize) throws IOException {
        // initialize output names along with action tuple size
        this.actionTupleSize = actionTupleSize;

        // we may need this logic for Ray 0.9.0
        List<String> tempOutputNames = new ArrayList<>();
//        if (actionTupleSize == 1) {
//            tempOutputNames.add("actions");
//        } else {
//            for (int i = 0; i < actionTupleSize; i++) {
//                tempOutputNames.add("actions_" + i);
//            }
//        }

        for (int i = 0; i < actionTupleSize; i++) {
            tempOutputNames.add("actions_" + i);
        }

        tempOutputNames.addAll(Arrays.asList(new String[]{"action_prob", "action_dist_inputs", "vf_preds", "action_logp"}));
        outputNames = tempOutputNames.toArray(new String[tempOutputNames.size()]);

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
        inputTensorShapes = new TensorShape[inputNames.length];
        for (int i = 0; i < inputNames.length; i++) {
            realInputNames[i] = signature.inputs().get(new BytePointer(inputNames[i])).name().getString();
            TensorShapeProto tsp = signature.inputs().get(new BytePointer(inputNames[i])).mutable_tensor_shape();
            if (tsp.dim_size() > 0) {
                TensorShapeProto_Dim dim = tsp.mutable_dim(0);
                if (dim.size() < 0) {
                    // use minibatch sizes of 1
                    dim.set_size(1);
                }
            }
            inputTensorShapes[i] = new TensorShape(tsp);
        }
        realOutputNames = new String[outputNames.length];
        for (int i = 0; i < outputNames.length; i++) {
            realOutputNames[i] = signature.outputs().get(new BytePointer(outputNames[i])).name().getString();
        }

        inputTensors = new Tensor[] {
                new Tensor(DT_FLOAT, inputTensorShapes[0]),
                Tensor.create(new long[1]),
                Tensor.create(new float[1]),
                Tensor.create(new boolean[1]),
                Tensor.create(new int[1])};
        obsData = new FloatPointer(inputTensors[0].tensor_data());

        outputTensors = new TensorVector[actionTupleSize];
        for (int i = 0; i < actionTupleSize; i++) {
            outputTensors[i] = new TensorVector(outputNames.length);
        }
    }

    @Override public float[] computeContinuousAction(float[] state) {
        if (disablePolicyHelper) {
            return null;
        }
        throw new UnsupportedOperationException();
    }

    @Override public long[] computeDiscreteAction(float[] state) {
        if (disablePolicyHelper) {
            return null;
        }
        obsData.put(state);

        long[] actionArray = new long[actionTupleSize];
        for (int i = 0; i < actionArray.length; i++) {
            Status s = bundle.session().Run(new StringTensorPairVector(realInputNames, inputTensors),
                    new StringVector(realOutputNames), new StringVector(), outputTensors[i]);
            if (!s.ok()) {
                throw new RuntimeException(s.error_message().getString());
            }
            actionArray[i] = new LongPointer(outputTensors[i].get(0).tensor_data()).get();
        }
//        for (int i = 0; i < outputTensors.size(); i++) {
//            System.out.println(outputTensors.get(i).createIndexer());
//        }
        return actionArray;

    }

}
