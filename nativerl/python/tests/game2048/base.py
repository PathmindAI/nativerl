import typing
import random
import time
import math

from . import logic
from . import constants as c


class Game2048:

    action: float = None  # Dynamically generated for each state by Pathmind

    scores = ["0", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1024"]

    number_of_actions: int = 4  # Needs to be provided
    grid_cells = 4 * 4
    number_of_observations: int = grid_cells * len(scores)  # Needs to be provided
    number_of_metrics: int = 2

    obs = dict.fromkeys(scores, [0] * grid_cells)

    reward_terms = {
        "log_game_score": 0,
        "invalid_move": 0,
    }
    prev_rew_terms = reward_terms
    total_reward = 0
    steps = 0

    def __init__(self, random_movements=False, human=False):
        self.random = random_movements
        self.human = human
        self.matrix = logic.new_game(c.GRID_LEN)

        self.commands = {'0': logic.up, '1': logic.down,
                         '2': logic.left, '3': logic.right}

        self.done = False
        self.obs = self.get_observation()
        self.reward_terms = self.get_reward()

    def reset(self) -> None:
        self.matrix = logic.new_game(c.GRID_LEN)
        self.steps = 0
        self.total_reward = 0
        self.done = False
        self.obs = self.get_observation()
        self.reward_terms = self.get_reward()

    def step(self) -> None:

        self.prev_rew_terms = self.reward_terms

        self.steps += 1
        if self.human and self.random:
            time.sleep(0.01)
        action = int(self.action) if not self.random else random.randint(0, 3)
        if self.human:
            print(f'Action: {["up", "down", "left", "right"][action]}')
        self.matrix, valid, rew = self.commands[str(action)](self.matrix)

        self.reward_terms["log_game_score"] = (math.log(rew, 2) if rew > 0 else 0) - self.prev_rew_terms["log_game_score"]
        if valid:
            self.matrix = logic.add_two(self.matrix)
        else:
            self.reward_terms["invalid_move"] = -1 - self.prev_rew_terms["invalid_move"]



        self.total_reward += sum(self.reward_terms.values())

    def get_observation(self) -> typing.Dict:
        for score in self.scores:
            self.obs[score] = [1 if value == score else 0 for row in self.matrix for value in row]

        return self.obs

    def get_reward(self) -> typing.Dict:
        return self.reward_terms

    def is_done(self) -> bool:
        return logic.game_state(self.matrix) in ['win', 'lose']

    def get_metrics(self) -> typing.List[float]:
        return [float(self.steps), float(self.total_reward)]
