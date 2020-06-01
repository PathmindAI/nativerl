package ai.skymind.nativerl;

import java.io.File;
import java.io.IOException;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;

import org.bytedeco.tensorflow.*;
import static org.bytedeco.tensorflow.global.tensorflow.*;

public class RLlibPolicyHelper implements PolicyHelper {
    static final String[] inputNames = {"observations", "prev_action", "prev_reward", "is_training", "seq_lens"};
    static final String[] outputNames = {"actions_0", "action_prob", "action_dist_inputs", "vf_preds", "action_logp"};

    SavedModelBundle bundle = null;
    SessionOptions options = null;
    RunOptions runOptions = null;
    StringUnorderedSet tags = null;
    TensorShape[] inputTensorShapes = null;
    String[] realInputNames = null;
    String[] realOutputNames = null;
    Tensor[] inputTensors = null;
    FloatPointer obsData = null;
    TensorVector outputTensors = null;

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
        outputTensors = new TensorVector(outputNames.length);
    }

    @Override public float[] computeContinuousAction(float[] state) {
        if (disablePolicyHelper) {
            return null;
        }
        throw new UnsupportedOperationException();
    }

    @Override public long computeDiscreteAction(float[] state) {
        if (disablePolicyHelper) {
            return -1;
        }
        obsData.put(state);
        Status s = bundle.session().Run(new StringTensorPairVector(realInputNames, inputTensors),
                new StringVector(realOutputNames), new StringVector(), outputTensors);
        if (!s.ok()) {
            throw new RuntimeException(s.error_message().getString());
        }
//        for (int i = 0; i < outputTensors.size(); i++) {
//            System.out.println(outputTensors.get(i).createIndexer());
//        }
        return new LongPointer(outputTensors.get(0).tensor_data()).get();
    }
}
