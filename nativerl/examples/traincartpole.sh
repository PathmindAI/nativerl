OUTPUT_DIR="$(pwd)"
MODEL_MODULE="cartpole"
ENVIRONMENT_CLASS="PathmindEnvironment"

cat <<EOF > $MODEL_MODULE.py
import math
import nativerl
import random

# based on: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
class $ENVIRONMENT_CLASS(nativerl.Environment):
    def __init__(self):
        nativerl.Environment.__init__(self)

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

    def getActionSpace(self, i):
        return nativerl.Discrete(2) if i == 0 else None

    def getActionMaskSpace(self):
        return None

    def getObservationSpace(self):
        return nativerl.Continuous(nativerl.FloatVector([-math.inf]), nativerl.FloatVector([math.inf]), nativerl.SSizeTVector([4]))

    def getMetricsSpace(self):
        return nativerl.Continuous(nativerl.FloatVector([-math.inf]), nativerl.FloatVector([math.inf]), nativerl.SSizeTVector([1]))

    def getNumberOfAgents(self):
        return 1

    def getActionMask(self, agentId):
        return None;

    def getObservation(self, agentId):
        return nativerl.Array(nativerl.FloatVector(self.state));

    def reset(self):
        self.state = [random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05)]
        self.steps = 0
        self.steps_beyond_done = None

    def setNextAction(self, action, agentId):
        self.action = action.values()[0]

    def step(self):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if self.action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = [x, x_dot, theta, theta_dot]
        self.steps += 1

    def isSkip(self, agentId):
        return False

    def isDone(self, agentId):
        x, x_dot, theta, theta_dot = self.state
        return bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
            or self.steps > 1000
        )

    def getReward(self, agentId):
        if not self.isDone(agentId):
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            self.steps_beyond_done += 1
            reward = 0.0
        return reward

    def getMetrics(self, agentId):
        return nativerl.Array(nativerl.FloatVector([] if self.steps_beyond_done is None else [self.steps_beyond_done]));
EOF

export CLASSPATH=$(find . -iname '*.jar' | tr '\n' :)


if which cygpath; then
    export CLASSPATH=$(cygpath --path --windows "$CLASSPATH")
    export PATH=$PATH:$(find "$(cygpath "$JAVA_HOME")" -name 'jvm.dll' -printf '%h:')
fi

PYTHON=$(which python.exe) || PYTHON=$(which python3)

"$PYTHON" run.py training \
    --algorithm "PPO" \
    --output-dir "$OUTPUT_DIR" \
    --environment "$MODEL_MODULE.$ENVIRONMENT_CLASS" \
    --num-workers 4 \
    --max-iterations 10 \
    --multi-agent
