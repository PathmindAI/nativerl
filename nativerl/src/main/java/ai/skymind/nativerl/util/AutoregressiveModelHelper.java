package ai.skymind.nativerl.util;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class AutoregressiveModelHelper {

    public static void main(String[] args) {
        System.out.println(AutoregressiveModelHelper.generateAutoregressiveClass(4, 4));
    }

    private static String sample(int actionTupleSize) {
        StringBuffer sb = new StringBuffer();
        sb.append("    def sample(self):\n");
        for (int i = 0; i < actionTupleSize; i++) {
            sb.append("        a" + (i+1) + "_dist = self._a" + (i+1) + "_distribution(" + (i > 0 ?  String.join(", ", formatNCopiesExclude("a%d", i)) : "") + ")\n");
            sb.append("        a" + (i+1) + " = a" + (i+1) + "_dist.sample()\n");

            if (i > 0) {
                sb.append("        self._action_logp = " + String.join(" + ", formatNCopies("a%d_dist.logp(a%d)", i)) + "\n");
            }
            sb.append("\n");
        }
        sb.append("        return TupleActions([" + String.join(", ", formatNCopiesExclude("a%d", actionTupleSize)) + "])\n\n");

        return sb.toString();
    }

    private static String deterministic_sample(int actionTupleSize) {
        StringBuffer sb = new StringBuffer();
        sb.append("    def deterministic_sample(self):\n");
        for (int i = 0; i < actionTupleSize; i++) {
            sb.append("        a" + (i+1) + "_dist = self._a" + (i+1) + "_distribution(" + (i > 0 ?  String.join(", ", formatNCopiesExclude("a%d", i)) : "") + ")\n");
            sb.append("        a" + (i+1) + " = a" + (i+1) + "_dist.sample()\n");

            if (i > 0) {
                sb.append("        self._action_logp = " + String.join(" + ", formatNCopies("a%d_dist.logp(a%d)", i)) + "\n");
            }
            sb.append("\n");
        }
        sb.append("        return TupleActions([" + String.join(", ", formatNCopiesExclude("a%d", actionTupleSize)) + "])\n\n");

        return sb.toString();
    }

    private static String logp(int actionTupleSize) {
        StringBuffer sb = new StringBuffer();
        sb.append("    def logp(self, actions):\n");
        sb.append("        " + String.join(", ", formatNCopiesExclude("a%d", actionTupleSize)) + "  = " + String.join(", ", formatNCopiesExcludeZeroIndex("actions[:, %d]", actionTupleSize)) + "\n");

        for (int i = 0; i < actionTupleSize; i++) {
            if (i > 0) {
                sb.append("        a" + i + "_vec = tf.expand_dims(tf.cast(a" + i + ", tf.float32), 1)\n");
            }
        }

        sb.append(
                "        " + String.join(", ", formatNCopiesExclude("a%d_logits", actionTupleSize))
                        + " = self.model.action_model([self.inputs, " + String.join(", ", formatNCopiesExclude("a%d_vec", actionTupleSize-1)) + "])\n"
        );

        sb.append(
                "        return (" + String.join(" + ", formatNCopiesExclude("Categorical(a%d_logits).logp(a%d)", actionTupleSize)) + ")\n\n"

        );

        return sb.toString();
    }

    private static String entropy(int actionTupleSize) {
        StringBuffer sb = new StringBuffer();
        sb.append("    def entropy(self):\n");

        for (int i = 0; i < actionTupleSize; i++) {
            sb.append("        " + "a" + (i+1) + "_dist = self._a" + (i+1) +"_distribution(" + (i > 0 ? String.join(", ", formatNCopiesExclude("a%d_dist.sample()", i)): "") + ")\n");
        }

        sb.append("        return " + String.join(" + ", formatNCopiesExclude("a%d_dist.entropy()", actionTupleSize)) + "\n\n");

        return sb.toString();
    }

    private static String kl(int actionTupleSize) {
        StringBuffer sb = new StringBuffer();
        sb.append("    def kl(self, other):\n");

        for (int i = 0; i < actionTupleSize; i++) {
            sb.append("        " + "a" + (i+1) + "_dist = self._a" + (i+1) +"_distribution(" + (i > 0 ? String.join(", ", formatNCopiesExclude("a%d", i)): "")+ ")\n");
            sb.append("        a" + (i+1) +"_terms = a" + (i+1) +"_dist.kl(other._a" + (i+1) +"_distribution(" + (i > 0 ? String.join(", ", formatNCopiesExclude("a%d", i)): "") + "))\n");
            sb.append((i == actionTupleSize-1 ? "\n" : "        a" + (i+1) +" = a" + (i+1) +"_dist.sample()\n\n"));
        }

        sb.append("        return " + String.join(" + ", formatNCopiesExclude("a%d_terms", actionTupleSize)) + "\n\n");

        return sb.toString();
    }

    private static String aN_distribution(int tupleActionSize) {
        StringBuffer sb = new StringBuffer();

        for (int i = 0; i < tupleActionSize; i++) {
            sb.append("    def _a" + (i+1) + "_distribution(self" + (i>0 ? String.join("", formatNCopiesExclude(", a%d", i)): "") + "):\n");
            if (i > 0) {
                sb.append(
                        String.join("", formatNCopiesExclude("        a%d_vec = tf.expand_dims(tf.cast(a%d, tf.float32), 1)\n", i))
                );
            }
            sb.append((i == tupleActionSize-1 ? "" : "        BATCH = tf.shape(self.inputs)[0]\n"));

            List<String> list = new ArrayList<>(Collections.nCopies(tupleActionSize, "_"));
            list.set(i, "a" + (i+1) + "_logits");
            List<String> list2 = new ArrayList<>(Collections.nCopies(tupleActionSize-1, "tf.zeros((BATCH, 1))"));
            if (i > 0) {
                for (int j = 0; j < i; j++) {
                    list2.set(j, "a" + (j + 1) + "_vec");
                }
            }

            sb.append("        " + String.join(", ", list) + " = self.model.action_model(\n");
            sb.append("            [self.inputs, " + String.join(", ", list2) + "])\n");
            sb.append("        a" + (i+1) + "_dist = Categorical(a" + (i+1) + "_logits)\n");
            sb.append("        return a" + (i+1) + "_dist\n");
            sb.append("\n");
        }

        return sb.toString();
    }

    private static List<String> formatNCopies(String format, int n) {
        List<String> list = new ArrayList<>();
        for (int j = 0; j <= n; j++) {
            list.add(String.format(format, j+1, j+1));
        }
        return list;
    }

    private static List<String> formatNCopiesExclude(String format, int n) {
        List<String> list = new ArrayList<>();
        for (int j = 0; j < n; j++) {
            list.add(String.format(format, j+1, j+1));
        }
        return list;
    }

    private static List<String> formatNCopiesExcludeZeroIndex(String format, int n) {
        List<String> list = new ArrayList<>();
        for (int j = 0; j < n; j++) {
            list.add(String.format(format, j, j));
        }
        return list;
    }

    private static String generateNaryAutoregressiveOutput(int actionTupleSize) {
        StringBuffer model = new StringBuffer();
        model.append("class NaryAutoregressiveOutput(ActionDistribution):\n");
        model.append("    \"\"\"Action distribution P(" + String.join(", ", formatNCopiesExclude("a%d", actionTupleSize)) + ") = ");
        for (int i = 0; i < actionTupleSize; i++) {
            if (i == 0 ) {
                model.append("P(a1)");
            } else {
                model.append(" * P(a" + (i+1) + " | " + String.join(", ", formatNCopiesExclude("a%d", i)) + ")");
            }
        }
        model.append("\"\"\"\n\n");

        model.append("    @staticmethod\n");
        model.append("    def required_model_output_shape(self, model_config):\n");
        model.append("        return " + (actionTupleSize > 2 ? 256 : 16) + "  # controls model output feature vector size\n\n");
        model.append(deterministic_sample(actionTupleSize));
        model.append(sample(actionTupleSize));
        model.append(logp(actionTupleSize));
        model.append("    def sampled_action_logp(self):\n");
        model.append("        return tf.exp(self._action_logp)\n\n");
        model.append(entropy(actionTupleSize));
        model.append(kl(actionTupleSize));
        model.append(aN_distribution(actionTupleSize));
        return model.toString();
    }

    public static String generateAutoregressiveActionsModel(int actionTupleSize, long numAction) {
        StringBuffer model = new StringBuffer();
        model.append("class AutoregressiveActionsModel(TFModelV2):\n");
        model.append("    \"\"\"Implements the `.action_model` branch required above.\"\"\"\n");
        model.append("    def __init__(self, obs_space, action_space, num_outputs, model_config,\n");
        model.append("                 name):\n");
        model.append("        super(AutoregressiveActionsModel, self).__init__(\n");
        model.append("            obs_space, action_space, num_outputs, model_config, name)\n");
        model.append("        if action_space != Tuple([" + String.join(", ", Collections.nCopies(actionTupleSize, "Discrete(" + numAction + ")")) + "]):\n");
        model.append("            raise ValueError(\n");
        model.append("                \"This model only supports the [" + String.join(", ", Collections.nCopies(actionTupleSize, String.valueOf(numAction))) + "] action space\")\n\n");
        model.append("        # Inputs\n");
        model.append("        obs_input = tf.keras.layers.Input(\n");
        model.append("            shape=obs_space.shape, name=\"obs_input\")\n");
        model.append("        ctx_input = tf.keras.layers.Input(\n");
        model.append("            shape=(num_outputs, ), name=\"ctx_input\")\n");

        for (int i = 0; i < actionTupleSize - 1; i++) {
            model.append("        a" + (i+1) + "_input = tf.keras.layers.Input(shape=(1, ), name=\"a" + (i+1) + "_input\")\n");
        }

        model.append("\n        # Output of the model (normally 'logits', but for an autoregressive\n");
        model.append("        # dist this is more like a context/feature layer encoding the obs)\n");
        model.append("        hidden_layer = tf.keras.layers.Dense(\n");
        model.append("            256, # hyperparameter choice\n");
        model.append("            name=\"hidden_1\",\n");
        model.append("            activation=tf.nn.tanh,\n");
        model.append("            kernel_initializer=normc_initializer(1.0))(obs_input)\n\n");
        model.append("        context = tf.keras.layers.Dense(\n");
        model.append("            num_outputs,\n");
        model.append("            name=\"hidden_2\",\n");
        model.append("            activation=tf.nn.tanh,\n");
        model.append("            kernel_initializer=normc_initializer(1.0))(hidden_layer)\n\n");
        model.append("        # V(s)\n");
        model.append("        value_out = tf.keras.layers.Dense(\n");
        model.append("            1,\n");
        model.append("            name=\"value_out\",\n");
        model.append("            activation=None,\n");
        model.append("            kernel_initializer=normc_initializer(0.01))(context)\n\n");

        for (int i = 0; i < actionTupleSize; i++) {
            if (i == 0) {
                model.append("        # P(a1 | obs)\n");
                model.append("        a1_logits = tf.keras.layers.Dense(\n");
                model.append("            " + numAction + ",\n");
                model.append("            name=\"a1_logits\",\n");
                model.append("            activation=None,\n");
                model.append("            kernel_initializer=normc_initializer(0.01))(ctx_input)\n\n");
            } else {
                model.append("        # P(a" + (i+1) + " | " + String.join(", ", formatNCopiesExclude("a%d", i)) + ")\n");
//                        "        a2_context = a1_input\n" +
                model.append("        a" + (i+1) + "_context = tf.keras.layers.Concatenate(axis=1)(\n");
                model.append("            [" + (i>1 ? "a" + i + "_context" : "ctx_input") + ", a" + i + "_input])\n");
                model.append("        a" + (i+1) + "_hidden = tf.keras.layers.Dense(\n");
                model.append("            64,\n");
                model.append("            name=\"a" + (i+1) + "_hidden\",\n");
                model.append("            activation=tf.nn.tanh,\n");
                model.append("            kernel_initializer=normc_initializer(1.0))(a" + (i+1) + "_context)\n");
                model.append("        a" + (i+1) + "_logits = tf.keras.layers.Dense(\n");
                model.append("            " + numAction + ",\n");
                model.append("            name=\"a" + (i+1) + "_logits\",\n");
                model.append("            activation=None,\n");
                model.append("            kernel_initializer=normc_initializer(0.01))(a" + (i+1) + "_hidden)\n\n");
            }
        }

        model.append("        # Base layers\n");
        model.append("        self.base_model = tf.keras.Model(obs_input, [context, value_out])\n");
        model.append("        self.register_variables(self.base_model.variables)\n");
        model.append("        self.base_model.summary()\n\n");
        model.append("        # Autoregressive action sampler\n");
        model.append("        self.action_model = tf.keras.Model([ctx_input" + String.join("",formatNCopiesExclude(", a%d_input", actionTupleSize-1)) + "],\n");
        model.append("                                           [" + String.join(", ", formatNCopiesExclude("a%d_logits", actionTupleSize)) + "])\n");
        model.append("        self.action_model.summary()\n");
        model.append("        self.register_variables(self.action_model.variables)\n\n");
        model.append("    def forward(self, input_dict, state, seq_lens):\n");
        model.append("        context, self._value_out = self.base_model(input_dict[\"obs\"])\n");
        model.append("        return context, state\n\n");
        model.append("    def value_function(self):\n");
        model.append("        return tf.reshape(self._value_out, [-1])\n\n");
        model.append("ModelCatalog.register_custom_model(\"autoregressive_model\",\n");
        model.append("                                   AutoregressiveActionsModel)\n");
        model.append("ModelCatalog.register_custom_action_dist(\"nary_autoreg_output\",\n");
        model.append("                                         NaryAutoregressiveOutput)\n\n");

        return model.toString();
    }

    public static String generateAutoregressiveClass(int actionTupleSize, long numAction) {
        return generateNaryAutoregressiveOutput(actionTupleSize) + "\n" + generateAutoregressiveActionsModel(actionTupleSize, numAction);
    }
}
