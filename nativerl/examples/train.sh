source setup.sh

MODEL_TYPE="ANYLOGIC"
if [[ "$ENVIRONMENT_NAME" ]]; then
    MODEL_TYPE="PYTHON"
    export ENVIRONMENT_CLASS=$ENVIRONMENT_NAME
fi

CPU_COUNT=$(getconf _NPROCESSORS_ONLN)

if [[ -z "$NUM_WORKERS" ]]; then
    if [[ $CPU_COUNT = 72 ]]; then
        export NUM_WORKERS=4
        export NUM_CPUS=2
    elif [[ $CPU_COUNT = 36 ]]; then
        export NUM_WORKERS=4
        export NUM_CPUS=2
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

if [[ -z "$NUM_SAMPLES" ]]; then
    if [[ $CPU_COUNT = 72 ]]; then
        export NUM_SAMPLES=8
    else
        export NUM_SAMPLES=4
    fi
fi

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

ACTION_MASKING_PARAM=""
if [[ "$ACTIONMASKS" = true ]]; then
    ACTION_MASKING_PARAM="--action-masking"
fi

FREEZING_PARAM=""
if [[ "$FREEZING" = true ]]; then
    FREEZING_PARAM="--freezing"
fi

CONVERGENCE_CHECK_START_ITERATION_PARAM=""
if [[ ! -z "$CONVERGENCE_CHECK_START_ITERATION" ]]; then
    CONVERGENCE_CHECK_START_ITERATION_PARAM="--convergence_check_start_iteration ${CONVERGENCE_CHECK_START_ITERATION}"
fi

mainAgent=$MAIN_AGENT
experimentClass=$EXPERIMENT_CLASS
EXPERIMENT_TYPE=$EXPERIMENT_TYPE

if [[ -z "$mainAgent" ]]; then
    mainAgent="Main"
fi
if [[ -z "$experimentClass" ]]; then
    experimentClass="Simulation"
fi
if [[ -z "$EXPERIMENT_TYPE" ]]; then
    EXPERIMENT_TYPE="Simulation"
fi

IS_GYM_PARAM=""
if [[ "$IS_GYM" = true ]]; then
    IS_GYM_PARAM="--is_gym"
fi

IS_PATHMIND_SIMULATION_PARAM=""
if [[ "$IS_PATHMIND_SIMULATION" = true ]]; then
    IS_PATHMIND_SIMULATION_PARAM="--is_pathmind_simulation"
fi

OBS_SELECTION_PARAM=""
if [[ ! -z "$OBS_SELECTION" ]]; then
    OBS_SELECTION_PARAM="--obs_selection $OBS_SELECTION"
fi

REW_FCT_NAME_PARAM=""
if [[ ! -z "$REW_FCT_NAME" ]]; then
    REW_FCT_NAME_PARAM="--rew_fct_name $REW_FCT_NAME"
fi

SCHEDULER_PARAM=""
if [[ "$SCHEDULER" = "PB2" ]]; then
    SCHEDULER_PARAM="--scheduler $SCHEDULER"
fi

NUM_HIDDEN_LAYERS_PARAM=""
if [[ ! -z "$NUM_HIDDEN_LAYERS" ]]; then
    NUM_HIDDEN_LAYERS_PARAM="--num_hidden_layers $NUM_HIDDEN_LAYERS"
fi

NUM_HIDDEN_NODES_PARAM=""
if [[ ! -z "$NUM_HIDDEN_NODES" ]]; then
    NUM_HIDDEN_NODES_PARAM="--num_hidden_nodes $NUM_HIDDEN_NODES"
fi

GAMMA_PARAM=""
if [[ ! -z "$GAMMA" ]]; then
    GAMMA_PARAM="--gamma $GAMMA"
fi

TRAIN_BATCH_MODE_PARAM=""
if [[ ! -z "$TRAIN_BATCH_MODE" ]]; then
    TRAIN_BATCH_MODE_PARAM="--train_batch_mode $TRAIN_BATCH_MODE"
fi

TRAIN_BATCH_SIZE_PARAM=""
if [[ ! -z "$TRAIN_BATCH_SIZE" ]]; then
    TRAIN_BATCH_SIZE_PARAM="--train_batch_size $TRAIN_BATCH_SIZE"
fi

ROLLOUT_FRAGMENT_LENGTH_PARAM=""
if [[ ! -z "$ROLLOUT_FRAGMENT_LENGTH" ]]; then
    ROLLOUT_FRAGMENT_LENGTH_PARAM="--rollout_fragment_length $ROLLOUT_FRAGMENT_LENGTH"
fi

REWARD_TERMS_WEIGHTS_PARAM=""
if [[ ! -z "$REWARD_TERMS_WEIGHTS" ]]; then
    REWARD_TERMS_WEIGHTS_PARAM="--alphas $REWARD_TERMS_WEIGHTS"
fi

NUM_REWARD_TERMS_PARAM=""
if [[ ! -z "$NUM_REWARD_TERMS" ]]; then
    NUM_REWARD_TERMS_PARAM="--num_reward_terms  $NUM_REWARD_TERMS"
fi

REWARD_BALANCE_PERIOD_PARAM=""
if [[ ! -z "$REWARD_BALANCE_PERIOD" ]]; then
    REWARD_BALANCE_PERIOD_PARAM="--reward_balance_period $REWARD_BALANCE_PERIOD"
fi

USE_AUTO_NORM_PARAM=""
if [[ ! -z "$USE_AUTO_NORM" ]]; then
    USE_AUTO_NORM_PARAM="--use_auto_norm $USE_AUTO_NORM"
fi

export OUTPUT_DIR=$(pwd)

if [[ "$MODEL_TYPE" = "ANYLOGIC" ]]; then
    export MODEL_PACKAGE=$(for m in $(ls model.jar lib/model*.jar 2> /dev/null) ; do unzip -l $m | grep /${mainAgent}.class; done | awk '{print $4}' | xargs dirname)
    export MODEL_PACKAGE_NAME=$(echo ${MODEL_PACKAGE} | sed 's/\//\./g')
    export ENVIRONMENT_CLASS="$MODEL_PACKAGE_NAME.PathmindEnvironment"
    export AGENT_CLASS="$MODEL_PACKAGE_NAME.${mainAgent}"
    export SIMULATION_PACKAGE=$(for m in $(ls model.jar lib/model*.jar 2> /dev/null) ; do unzip -l $m | grep /${experimentClass}.class | grep -v pathmind/policyhelper; done | awk '{print $4}' | xargs dirname)
    export SIMULATION_PACKAGE_NAME=$(echo $SIMULATION_PACKAGE | sed 's/\//\./g')
    export SIMULATION_CLASS="$SIMULATION_PACKAGE_NAME.${experimentClass}"

    mkdir -p $MODEL_PACKAGE

    export CLASSPATH=$(find . -iname '*.jar' | tr '\n' :)

    if which cygpath; then
        export CLASSPATH=$(cygpath --path --windows "$CLASSPATH")
        export PATH=$PATH:$(find "$(cygpath "$JAVA_HOME")" -name 'jvm.dll' -printf '%h:')
    fi

    java ai.skymind.nativerl.AnyLogicHelper \
        --environment-class-name "$ENVIRONMENT_CLASS" \
        --simulation-class-name "$SIMULATION_CLASS" \
        --output-dir "$OUTPUT_DIR" \
        --algorithm "PPO" \
        --agent-class-name "$AGENT_CLASS" \
        --class-snippet "$CLASS_SNIPPET" \
        --reset-snippet "$RESET_SNIPPET" \
        --reward-terms-snippet "$REWARD_TERMS_SNIPPET" \
        --simulation-parameter-snippet "$SIMULATION_PARAMETER_SNIPPET" \
        --observation-snippet "$OBSERVATION_SNIPPET" \
        --metrics-snippet "$METRICS_SNIPPET" \
        --experiment-type "$EXPERIMENT_TYPE" \
        --test-iterations 0 \
        --policy-helper RLlibPolicyHelper \
        $NAMED_VARIABLE_PARAM \
        $MULTIAGENT_PARAM \

    java ai.skymind.nativerl.LearningAgentHelper

    javac $(find -iname '*.java')
fi

mkdir -p $OUTPUT_DIR/PPO
cp -r python/* .

PYTHON=$(which python.exe) || PYTHON=$(which python3)

"$PYTHON"  run.py training \
    --algorithm "PPO" \
    --output-dir "$OUTPUT_DIR" \
    --environment "$ENVIRONMENT_CLASS" \
    --num-cpus $NUM_CPUS \
    --num-workers $NUM_WORKERS \
    --max-iterations $MAX_ITERATIONS \
    --max-time-in-sec $MAX_TIME_IN_SEC \
    --num-samples $NUM_SAMPLES \
    --checkpoint-frequency $CHECKPOINT_FREQUENCY \
    --cpu-count $CPU_COUNT \
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
    $FREEZING_PARAM \
    $IS_GYM_PARAM \
    $IS_PATHMIND_SIMULATION_PARAM \
    $OBS_SELECTION_PARAM \
    $REW_FCT_NAME_PARAM \
    $SCHEDULER_PARAM \
    $CONVERGENCE_CHECK_START_ITERATION_PARAM \
    $NUM_HIDDEN_LAYERS_PARAM \
    $NUM_HIDDEN_NODES_PARAM \
    $GAMMA_PARAM \
    $TRAIN_BATCH_MODE_PARAM \
    $ROLLOUT_FRAGMENT_LENGTH_PARAM \
    $TRAIN_BATCH_SIZE_PARAM \
    $REWARD_TERMS_WEIGHTS_PARAM \
    $NUM_REWARD_TERMS_PARAM \
    $REWARD_BALANCE_PERIOD_PARAM \
    $USE_AUTO_NORM_PARAM
