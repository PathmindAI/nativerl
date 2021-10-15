OUTPUT_DIR="$(pwd)"
MODEL_PACKAGE="traffic_light_opt"
ENVIRONMENT_CLASS="$MODEL_PACKAGE.PathmindEnvironment"
SIMULATION_CLASS="$MODEL_PACKAGE.Simulation"
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

$REWARD_TERMS_SNIPPET='
    if (before == null) return 0;
    double[] s0 = before.vars, s1 = after.vars;
    // change in forward + intersection delay
    double delay0 = s0[0] + s0[2] + s0[4] + s0[6] + s0[8];
    double delay1 = s1[0] + s1[2] + s1[4] + s1[6] + s1[8];
    reward = delay0 - delay1;
    if (delay0 > 0 || delay1 > 0) {
        rewardTermsRaw[0] /= Math.max(delay0, delay1);
    }
'

METRICS_SNIPPET='
    metrics = new double[] { agent.tisDS.getYMean() };
'

mkdir -p $MODEL_PACKAGE

export CLASSPATH=$(find . -iname '*.jar' | tr '\n' :)

if which cygpath; then
    export CLASSPATH=$(cygpath --path --windows "$CLASSPATH")
    export PATH=$PATH:$(find "$(cygpath "$JAVA_HOME")" -name 'jvm.dll' -printf '%h:')
fi

java ai.skymind.nativerl.AnyLogicHelper \
    --environment-class-name "$ENVIRONMENT_CLASS" \
    --simulation-class-name "$SIMULATION_CLASS" \
    --agent-class-name "$AGENT_CLASS" \
    --class-snippet "$CLASS_SNIPPET" \
    --reset-snippet "$RESET_SNIPPET" \
    --observation-snippet "$OBSERVATION_SNIPPET" \
    --reward-terms-snippet "$REWARD_TERMS_SNIPPET" \
    --metrics-snippet "$METRICS_SNIPPET" \
    --policy-helper RLlibPolicyHelper \
    --multi-agent \
    --named-variables

javac $(find -iname '*.java')

PYTHON=$(which python.exe) || PYTHON=$(which python3)

"$PYTHON" run.py training \
    --algorithm "PPO" \
    --output-dir "$OUTPUT_DIR" \
    --environment "$ENVIRONMENT_CLASS" \
    --num-workers 4 \
    --random-seed 42 \
    --max-iterations 10 \
    --max-reward-mean 100 \
    --multi-agent \
    rllibtrain.py

# Execute the simulation with all models to get test metrics
#find "$OUTPUT_DIR" -iname model -type d -exec java "$ENVIRONMENT_CLASS" {} \;
