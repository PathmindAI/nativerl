OUTPUT_DIR="$(pwd)"
ENVIRONMENT_CLASS="traffic_light_opt.TrafficEnvironment"
AGENT_CLASS="traffic_light_opt.Main"

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

REWARD_SNIPPET='
    // change in forward + intersection delay
    double delay0 = state0[0] + state0[2] + state0[4] + state0[6] + state0[8];
    double delay1 = state1[0] + state1[2] + state1[4] + state1[6] + state1[8];
    reward = delay0 - delay1;
    if (delay0 > 0 || delay1 > 0) {
        reward /= Math.max(delay0, delay1);
    }
'

METRICS_SNIPPET='
    metrics = new double[] { agent.tisDS.getYMean() };
'

export CLASSPATH=$(find -iname '*.jar' -printf '%p:')

java ai.skymind.nativerl.AnyLogicHelper \
    --environment-class-name "$ENVIRONMENT_CLASS" \
    --agent-class-name "$AGENT_CLASS" \
    --discrete-actions 2 \
    --continuous-observations 10 \
    --step-time 10 \
    --stop-time 28800 \
    --random-seed 1 \
    --class-snippet "$CLASS_SNIPPET" \
    --reset-snippet "$RESET_SNIPPET" \
    --reward-snippet "$REWARD_SNIPPET" \
    --metrics-snippet "$METRICS_SNIPPET" \
    --policy-helper RLlibPolicyHelper

javac $(find -iname '*.java')

java ai.skymind.nativerl.RLlibHelper \
    --output-dir "$OUTPUT_DIR" \
    --environment "$ENVIRONMENT_CLASS" \
    --num-workers 4 \
    --random-seed 42 \
    rllibtrain.py

python3 rllibtrain.py

for MODEL in $(find $OUTPUT_DIR -iname model -type d); do
    java "$ENVIRONMENT_CLASS" "$MODEL"
end