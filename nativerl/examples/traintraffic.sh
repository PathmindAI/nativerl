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
    double reward = delay0 - delay1;
    if (delay0 > 0 || delay1 > 0) {
        reward /= Math.max(delay0, delay1);
    }
'

export CLASSPATH=$(find -iname '*.jar' -printf '%p:')

java ai.skymind.nativerl.AnyLogicHelper \
    --environment-class-name traffic_light_opt.TrafficEnvironment \
    --agent-class-name traffic_light_opt.Main \
    --discrete-actions 2 \
    --continuous-observations 10 \
    --step-time 10 \
    --stop-time 28800 \
    --random-seed 1 \
    --class-snippet "$CLASS_SNIPPET" \
    --reset-snippet "$RESET_SNIPPET" \
    --reward-snippet "$REWARD_SNIPPET"

javac $(find -iname '*.java')

java ai.skymind.nativerl.RLlibHelper \
    --environment traffic_light_opt.TrafficEnvironment \
    --num-workers 4 \
    --random-seed 42 \
    rllibtrain.py

python3 rllibtrain.py
