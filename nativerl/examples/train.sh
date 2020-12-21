source setup.sh

mainAgent=$MAIN_AGENT
experimentClass=$EXPERIMENT_CLASS
EXPERIMENT_TYPE=$EXPERIMENT_TYPE

if [[ -z "$mainAgent" ]]; then
    echo "main agent missing"
    mainAgent="Main"
fi
if [[ -z "$experimentClass" ]]; then
    experimentClass="Simulation"
fi
if [[ -z "$EXPERIMENT_TYPE" ]]; then
    EXPERIMENT_TYPE="Simulation"
fi

export MODEL_PACKAGE=$(for m in $(ls model.jar lib/model*.jar 2> /dev/null) ; do unzip -l $m | grep /${mainAgent}.class; done | awk '{print $4}' | xargs dirname)
export MODEL_PACKAGE_NAME=$(echo ${MODEL_PACKAGE} | sed 's/\//\./g')
export ENVIRONMENT_CLASS="$MODEL_PACKAGE_NAME.PathmindEnvironment"
export AGENT_CLASS="$MODEL_PACKAGE_NAME.${mainAgent}"
export SIMULATION_PACKAGE=$(for m in $(ls model.jar lib/model*.jar 2> /dev/null) ; do unzip -l $m | grep /${experimentClass}.class | grep -v pathmind/policyhelper; done | awk '{print $4}' | xargs dirname)
export SIMULATION_PACKAGE_NAME=$(echo $SIMULATION_PACKAGE | sed 's/\//\./g')
export SIMULATION_CLASS="$SIMULATION_PACKAGE_NAME.${experimentClass}"

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
    MULTIAGENT_PARAM="--multi_agent"
fi

DEBUGMETRICS_PARAM=""
if [[ "$DEBUGMETRICS" = true ]]; then
    DEBUGMETRICS_PARAM="--debug_metrics"
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
    USER_LOG_PARAM="--user_log"
fi

EPISODE_REWARD_RANGE_PARAM=""
if [[ ! -z "$EPISODE_REWARD_RANGE" ]]; then
    EPISODE_REWARD_RANGE_PARAM="--episode_reward_range ${EPISODE_REWARD_RANGE}"
fi

ENTROPY_SLOPE_PARAM=""
if [[ ! -z "$ENTROPY_SLOPE" ]]; then
    ENTROPY_SLOPE_PARAM="--entropy_slope ${ENTROPY_SLOPE}"
fi

VF_LOSS_RANGE_PARAM=""
if [[ ! -z "$VF_LOSS_RANGE" ]]; then
    VF_LOSS_RANGE_PARAM="--vf_loss_range ${VF_LOSS_RANGE}"
fi

VALUE_PRED_PARAM=""
if [[ ! -z "$VALUE_PRED" ]]; then
    VALUE_PRED_PARAM="--value_pred ${VALUE_PRED}"
fi

NAMED_VARIABLE_PARAM=""
if [[ "$NAMED_VARIABLE" = true ]]; then
    NAMED_VARIABLE_PARAM="--named-variables"
fi

MAX_MEMORY_IN_MB_PARAM=""
if [[ ! -z "$MAX_MEMORY_IN_MB" ]]; then
    MAX_MEMORY_IN_MB_PARAM="--max_memory_in_mb ${MAX_MEMORY_IN_MB}"
fi

ACTION_MASKING_PARAM=""
if [[ "$ACTIONMASKS" = true ]]; then
    ACTION_MASKING_PARAM="--action_masking"
fi

FREEZING_PARAM=""
if [[ "$FREEZING" = true ]]; then
    FREEZING_PARAM="--freezing"
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
    --experiment-type "$EXPERIMENT_TYPE" \
    --test-iterations 0 \
    --policy-helper RLlibPolicyHelper \
    $NAMED_VARIABLE_PARAM \
    $MULTIAGENT_PARAM \

java ai.skymind.nativerl.LearningAgentHelper

javac $(find -iname '*.java')

mkdir -p $OUTPUT_DIR/PPO
cp run.py $OUTPUT_DIR/PPO
cp -r pathmind $OUTPUT_DIR/PPO

python3 run.py training \
    --algorithm "PPO" \
    --output_dir "$OUTPUT_DIR" \
    --environment "$ENVIRONMENT_CLASS" \
    --num_cpus $NUM_CPUS \
    --num_workers $NUM_WORKERS \
    --max_iterations $MAX_ITERATIONS \
    --max_time_in_sec $MAX_TIME_IN_SEC \
    --num_samples $NUM_SAMPLES \
    --checkpoint_frequency $CHECKPOINT_FREQUENCY \
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
    $ACTION_MASKING_PARAM \
    $FREEZING_PARAM
