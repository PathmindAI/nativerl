OUTPUT_DIR="$(pwd)"
MODEL_PACKAGE="traffic_light_opt"
ENVIRONMENT_CLASS="$MODEL_PACKAGE.PathmindEnvironment"
AGENT_CLASS="$MODEL_PACKAGE.Main"

CLASS_SNIPPET='
    int simCount = 0;
    String combinations[][] = {
            {"constant_moderate", "constant_moderate"},
            {"none_til_heavy_afternoon_peak", "constant_moderate"},
            {"constant_moderate", "none_til_heavy_afternoon_peak"},
            {"peak_afternoon", "peak_morning"},
            {"peak_morning", "peak_afternoon"}
    };
'

RESET_SNIPPET='
    simCount++;
    agent.schedNameNS = combinations[simCount % combinations.length][0];
    agent.schedNameEW = combinations[simCount % combinations.length][1];
'

OBSERVATION_SNIPPET='
    out = in.obs;
'

REWARD_SNIPPET='
    double[] s0 = before.vars, s1 = after.vars;
    // change in forward + intersection delay
    double delay0 = s0[0] + s0[2] + s0[4] + s0[6] + s0[8];
    double delay1 = s1[0] + s1[2] + s1[4] + s1[6] + s1[8];
    reward = delay0 - delay1;
    if (delay0 > 0 || delay1 > 0) {
        reward /= Math.max(delay0, delay1);
    }
'

METRICS_SNIPPET='
    metrics = new double[] { agent.tisDS.getYMean() };
'

mkdir -p $MODEL_PACKAGE

cat <<EOF > $MODEL_PACKAGE/Training.java
package $MODEL_PACKAGE;
import com.anylogic.engine.AgentConstants;
import com.anylogic.engine.AnyLogicInternalCodegenAPI;
import com.anylogic.engine.Engine;
import com.anylogic.engine.ExperimentCustom;
import com.anylogic.engine.Utilities;

public class Training extends ExperimentCustom {
    @AnyLogicInternalCodegenAPI
    public static String[] COMMAND_LINE_ARGUMENTS_xjal = new String[0];

    public Training(Object parentExperiment) {
        super(parentExperiment);
        this.setCommandLineArguments_xjal(COMMAND_LINE_ARGUMENTS_xjal);
    }

    public void run() {
    }

    @AnyLogicInternalCodegenAPI
    public void setupEngine_xjal(Engine engine) {
        Simulation sim = new Simulation();
        sim.setupEngine(engine);
        sim.initDefaultRandomNumberGenerator(engine);
    }

    @AnyLogicInternalCodegenAPI
    public static void main(String[] args) {
        COMMAND_LINE_ARGUMENTS_xjal = args;
        Utilities.prepareBeforeExperimentStart_xjal(Training.class);
        Training ex = new Training((Object)null);
        ex.setCommandLineArguments_xjal(args);
        ex.run();
    }
}
EOF

export CLASSPATH=$(find -iname '*.jar' -printf '%p:')

java ai.skymind.nativerl.AnyLogicHelper \
    --environment-class-name "$ENVIRONMENT_CLASS" \
    --agent-class-name "$AGENT_CLASS" \
    --class-snippet "$CLASS_SNIPPET" \
    --reset-snippet "$RESET_SNIPPET" \
    --observation-snippet "$OBSERVATION_SNIPPET" \
    --reward-snippet "$REWARD_SNIPPET" \
    --metrics-snippet "$METRICS_SNIPPET" \
    --policy-helper RLlibPolicyHelper \
    --multi-agent \
    --named-variables

javac $(find -iname '*.java')

java ai.skymind.nativerl.RLlibHelper \
    --algorithm "PPO" \
    --output-dir "$OUTPUT_DIR" \
    --environment "$ENVIRONMENT_CLASS" \
    --num-workers 4 \
    --random-seed 42 \
    --max-reward-mean 100 \
    --multi-agent \
    rllibtrain.py

python3 rllibtrain.py

# Execute the simulation with all models to get test metrics
#find "$OUTPUT_DIR" -iname model -type d -exec java "$ENVIRONMENT_CLASS" {} \;
