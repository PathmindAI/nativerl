# Have you ever seen a more complicated way to massage a gym env into something
# else, only to make it a gym env again internally? You're welcome.

import os

if os.environ.get("USE_PY_NATIVERL"):
    import pathmind_training.pynativerl as nativerl
else:
    import nativerl

import math
import random

import numpy as np


class PathmindEnvironment(nativerl.Environment):
    def __init__(self):
        self.state = None
        self.steps = 0
        self.steps_beyond_done = None
        self.action = None

        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.total_mass = self.mass_pole + self.mass_cart
        self.length = 0.5  # actually half the pole's length
        self.pole_mass_length = self.mass_pole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

    def getActionSpace(self, agent_id=0):
        return nativerl.Discrete(n=2) if agent_id == 0 else None

    def getActionMaskSpace(self):
        return None

    def getObservationSpace(self):
        return nativerl.Continuous([-math.inf], [math.inf], [4])

    def getMetricsSpace(self):
        return nativerl.Continuous([-math.inf], [math.inf], [1])

    def getNumberOfAgents(self):
        return 1

    def getActionMask(self, agent_id=0):
        return None

    def getObservation(self, agent_id=0):
        return np.asarray(self.state)

    def reset(self):
        self.state = [
            random.uniform(-0.05, 0.05),
            random.uniform(-0.05, 0.05),
            random.uniform(-0.05, 0.05),
            random.uniform(-0.05, 0.05),
        ]
        self.steps = 0
        self.steps_beyond_done = None

    def setNextAction(self, action, agent_id=0):
        self.action = action

    def step(self):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if self.action == 1 else -self.force_mag
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        temp = (
            force + self.pole_mass_length * theta_dot ** 2 * sin_theta
        ) / self.total_mass
        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / (
            self.length
            * (4.0 / 3.0 - self.mass_pole * cos_theta ** 2 / self.total_mass)
        )
        x_acc = temp - self.pole_mass_length * theta_acc * cos_theta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * x_acc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * theta_acc

        self.state = [x, x_dot, theta, theta_dot]
        self.steps += 1

    def isSkip(self, agent_id=0):
        return False

    def isDone(self, agent_id=0):
        x, x_dot, theta, theta_dot = self.state
        return bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
            or self.steps > 1000
        )

    def getReward(self, agent_id=0):
        if not self.isDone(agent_id):
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            self.steps_beyond_done += 1
            reward = 0.0
        return reward

    def getRewardTerms(self):
        return {"reward": self.getReward()}

    def getMetrics(self, agent_id=0):
        return (
            np.asarray([self.steps_beyond_done])
            if self.steps_beyond_done
            else np.asarray([])
        )
