source setup.sh
export MODEL_PACKAGE=$(unzip -l model.jar | grep Main.class | awk '{print $4}' | xargs dirname)
export MODEL_PACKAGE_NAME=$(echo $MODEL_PACKAGE | sed 's/\//\./g')
export ENVIRONMENT_CLASS="$MODEL_PACKAGE_NAME.PathmindEnvironment"
export AGENT_CLASS="$MODEL_PACKAGE_NAME.Main"
export OUTPUT_DIR=$(pwd)

if [[ -z "$NUM_WORKERS" ]]; then
    CPU_COUNT=$(lscpu -p | egrep -v '^#' | wc -l)
    SAMPLES="${NUM_SAMPLES:-4}"
    let WORKERS=(CPU_COUNT/SAMPLES)-1
    export NUM_WORKERS=$WORKERS
fi

if [[ $NUM_WORKERS < 1 ]]; then
    export NUM_WORKERS=1
fi

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

AUTOREGRESSIVE_PARAM=""
if [[ "$AUTOREGRESSIVE" = true ]]; then
    AUTOREGRESSIVE_PARAM="--autoregressive"
fi

RESUME_PARAM=""
if [[ "$RESUME" = true ]]; then
    RESUME_PARAM="--resume"
fi

USER_LOG_PARAM=""
if [[ "$USER_LOG" = true ]]; then
    USER_LOG_PARAM="--user-log"
fi

EPISODE_REWARD_RANGE_PARAM=""
if [[ ! -z "$EPISODE_REWARD_RANGE" ]]; then
    EPISODE_REWARD_RANGE_PARAM="--episode-reward-range ${EPISODE_REWARD_RANGE}"
fi

ENTROPY_SLOPE_PARAM=""
if [[ ! -z "$ENTROPY_SLOPE" ]]; then
    ENTROPY_SLOPE_PARAM="--entropy-slope ${ENTROPY_SLOPE}"
fi

VF_LOSS_RANGE_PARAM=""
if [[ ! -z "$VF_LOSS_RANGE" ]]; then
    VF_LOSS_RANGE_PARAM="--vf-loss-range ${VF_LOSS_RANGE}"
fi

VALUE_PRED_PARAM=""
if [[ ! -z "$VALUE_PRED" ]]; then
    VALUE_PRED_PARAM="--value-pred ${VALUE_PRED}"
fi

export CLASSPATH=$(find -iname '*.jar' -printf '%p:')

java ai.skymind.nativerl.AnyLogicHelper \
    --environment-class-name "$ENVIRONMENT_CLASS" \
    --agent-class-name "$AGENT_CLASS" \
    --discrete-actions $DISCRETE_ACTIONS \
    --action-tuple-size $ACTION_TUPLE_SIZE \
    --continuous-observations $CONTINUOUS_OBSERVATIONS \
    --step-time $STEP_TIME \
    --stop-time $STOP_TIME \
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
    --max-iterations $MAX_ITERATIONS \
    --max-time-in-sec $MAX_TIME_IN_SEC \
    --num-samples $NUM_SAMPLES \
    --discrete-actions $DISCRETE_ACTIONS \
    --action-tuple-size $ACTION_TUPLE_SIZE \
    --checkpoint-frequency $CHECKPOINT_FREQUENCY \
    $RESUME_PARAM \
    $AUTOREGRESSIVE_PARAM \
    $MULTIAGENT_PARAM \
    $EPISODE_REWARD_RANGE_PARAM \
    $ENTROPY_SLOPE_PARAM \
    $VF_LOSS_RANGE_PARAM \
    $VALUE_PRED_PARAM \
    $USER_LOG_PARAM \
    rllibtrain.py

mkdir -p $OUTPUT_DIR/PPO
cp rllibtrain.py $OUTPUT_DIR/PPO

set -e
if [[ "$RESUME" = true ]]; then
    mv examples/pm_resume.py .
    python3 pm_resume.py
fi
python3 rllibtrain.py
