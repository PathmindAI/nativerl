source setup.sh
export MODEL_PACKAGE=$(for m in $(ls model.jar lib/model*.jar 2> /dev/null) ; do unzip -l $m | grep Main.class; done | awk '{print $4}' | xargs dirname)
export MODEL_PACKAGE_NAME=$(echo $MODEL_PACKAGE | sed 's/\//\./g')
export ENVIRONMENT_CLASS="$MODEL_PACKAGE_NAME.PathmindEnvironment"

EXPERIMENT_TYPE="Simulation"
export SIMULATION_PACKAGE=$(for m in $(ls model.jar lib/model*.jar 2> /dev/null) ; do unzip -l $m | grep Simulation.class | grep -v pathmind/policyhelper; done | awk '{print $4}' | xargs dirname)
if [[ -z "$SIMULATION_PACKAGE" ]]; then
    export SIMULATION_PACKAGE=$(for m in $(ls model.jar lib/model*.jar 2> /dev/null) ; do unzip -l $m | grep RLExperiment.class | grep -v pathmind/policyhelper; done | awk '{print $4}' | xargs dirname)
    EXPERIMENT_TYPE="RLExperiment"
fi
export SIMULATION_PACKAGE_NAME=$(echo $SIMULATION_PACKAGE | sed 's/\//\./g')
export SIMULATION_CLASS="$SIMULATION_PACKAGE_NAME.$EXPERIMENT_TYPE"
export AGENT_CLASS="$MODEL_PACKAGE_NAME.Main"
export OUTPUT_DIR=$(pwd)

if [[ -z "$NUM_WORKERS" ]]; then
    CPU_COUNT=$(lscpu -p | egrep -v '^#' | wc -l)
    if [[ $CPU_COUNT = 36 ]]; then
        export NUM_WORKERS=2
        export NUM_CPUS=4
    elif [[ $CPU_COUNT = 16 ]]; then
        export NUM_WORKERS=3
        export NUM_CPUS=1
    else
        export NUM_WORKERS=1
        export NUM_CPUS=1
    fi
fi

if [[ $NUM_WORKERS < 1 ]]; then
    export NUM_WORKERS=1
fi

if [[ $NUM_CPUS < 1 ]]; then
    export NUM_CPUS=1
fi

mkdir -p $MODEL_PACKAGE

MULTIAGENT_PARAM=""
if [[ "$MULTIAGENT" = true ]]; then
    MULTIAGENT_PARAM="--multi-agent"
fi

DEBUGMETRICS_PARAM=""
if [[ "$DEBUGMETRICS" = true ]]; then
    DEBUGMETRICS_PARAM="--debug-metrics"
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

NAMED_VARIABLE_PARAM=""
if [[ "$NAMED_VARIABLE" = true ]]; then
    NAMED_VARIABLE_PARAM="--named-variables"
fi

MAX_MEMORY_IN_MB_PARAM=""
if [[ ! -z "$MAX_MEMORY_IN_MB" ]]; then
    MAX_MEMORY_IN_MB_PARAM="--max-memory-in-mb ${MAX_MEMORY_IN_MB}"
fi

export CLASSPATH=$(find . -iname '*.jar' | tr '\n' :)

java ai.skymind.nativerl.AnyLogicHelper \
    --environment-class-name "$ENVIRONMENT_CLASS" \
    --simulation-class-name "$SIMULATION_CLASS" \
    --output-dir "$OUTPUT_DIR" \
    --algorithm "PPO" \
    --agent-class-name "$AGENT_CLASS" \
    --class-snippet "$CLASS_SNIPPET" \
    --reset-snippet "$RESET_SNIPPET" \
    --reward-snippet "$REWARD_SNIPPET" \
    --observation-snippet "$OBSERVATION_SNIPPET" \
    --metrics-snippet "$METRICS_SNIPPET" \
    --test-iterations 0 \
    --policy-helper RLlibPolicyHelper \
    $NAMED_VARIABLE_PARAM \
    $MULTIAGENT_PARAM \

java ai.skymind.nativerl.LearningAgentHelper

javac $(find -iname '*.java')

# CHECKPOINT_PARAM=""
# if [[ ! -z "$CHECKPOINT" ]]; then
#     CHECKPOINT_PARAM="--checkpoint $CHECKPOINT"
# fi

java ai.skymind.nativerl.RLlibHelper \
    --algorithm "PPO" \
    --output-dir "$OUTPUT_DIR" \
    --environment "$ENVIRONMENT_CLASS" \
    --num-cpus $NUM_CPUS \
    --num-workers $NUM_WORKERS \
    --max-iterations $MAX_ITERATIONS \
    --max-time-in-sec $MAX_TIME_IN_SEC \
    --num-samples $NUM_SAMPLES \
    --checkpoint-frequency $CHECKPOINT_FREQUENCY \
    $RESUME_PARAM \
    $AUTOREGRESSIVE_PARAM \
    $MULTIAGENT_PARAM \
    $DEBUGMETRICS_PARAM \
    $EPISODE_REWARD_RANGE_PARAM \
    $ENTROPY_SLOPE_PARAM \
    $VF_LOSS_RANGE_PARAM \
    $VALUE_PRED_PARAM \
    $USER_LOG_PARAM \
    $MAX_MEMORY_IN_MB_PARAM \
    rllibtrain.py

mkdir -p $OUTPUT_DIR/PPO
cp rllibtrain.py $OUTPUT_DIR/PPO

python3 rllibtrain.py
