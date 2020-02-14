source setup.sh
export MODEL_PACKAGE=$(unzip -l model.jar | grep Main.class | awk '{print $4}' | xargs dirname)
export MODEL_PACKAGE_NAME=$(echo $MODEL_PACKAGE | sed 's/\//\./g')
export ENVIRONMENT_CLASS="$MODEL_PACKAGE_NAME.PathmindEnvironment"
export AGENT_CLASS="$MODEL_PACKAGE_NAME.Main"
PHYSICAL_CPU_COUNT=$(lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l)
let WORKERS=$PHYSICAL_CPU_COUNT-1
export NUM_WORKERS=$WORKERS
export OUTPUT_DIR=$(pwd)

mkdir -p $MODEL_PACKAGE

cat <<EOF > $MODEL_PACKAGE/Training.java
package $MODEL_PACKAGE_NAME;
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

MULTIAGENT_PARAM=""
if [[ "$MULTIAGENT" = true ]]; then
    MULTIAGENT_PARAM="--multi-agent"
fi

export CLASSPATH=$(find -iname '*.jar' -printf '%p:')

java ai.skymind.nativerl.AnyLogicHelper \
    --environment-class-name "$ENVIRONMENT_CLASS" \
    --agent-class-name "$AGENT_CLASS" \
    --discrete-actions $DISCRETE_ACTIONS \
    --continuous-observations $CONTINUOUS_OBSERVATIONS \
    --step-time $STEP_TIME \
    --stop-time $STOP_TIME \
    --random-seed $RANDOM_SEED \
    --class-snippet "$CLASS_SNIPPET" \
    --reset-snippet "$RESET_SNIPPET" \
    --reward-snippet "$REWARD_SNIPPET" \
    --metrics-snippet "$METRICS_SNIPPET" \
    --test-iterations 0 \
    --policy-helper RLlibPolicyHelper \
    $MULTIAGENT_PARAM \

javac $(find -iname '*.java')

# CHECKPOINT_PARAM=""
# if [[ ! -z "$CHECKPOINT" ]]; then
#     CHECKPOINT_PARAM="--checkpoint $CHECKPOINT"
# fi

java ai.skymind.nativerl.RLlibHelper \
    --algorithm "PPO" \
    --output-dir "$OUTPUT_DIR" \
    --environment "$ENVIRONMENT_CLASS" \
    --num-workers $NUM_WORKERS \
    --random-seed $RANDOM_SEED \
    --max-reward-mean $MAX_REWARD_MEAN \
    --max-iterations $MAX_ITERATIONS \
    --max-time-in-sec $MAX_TIME_IN_SEC \
    --num-samples $NUM_SAMPLES \
    $MULTIAGENT_PARAM \
    rllibtrain.py

python3 rllibtrain.py

