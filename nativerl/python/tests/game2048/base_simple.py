import random
import time
import typing

from . import constants as c
from . import logic


class Game2048:

    action: float = None  # Dynamically generated for each state by Pathmind

    number_of_actions: int = 4  # Needs to be provided
    number_of_observations: int = 176  # Needs to be provided
    number_of_metrics: int = 2

    score = 0
    rew = 0
    steps = 0

    def __init__(self, random_movements=False, human=False):
        self.random = random_movements
        self.human = human
        self.matrix = logic.new_game(c.GRID_LEN)

        self.commands = {
            "0": logic.up,
            "1": logic.down,
            "2": logic.left,
            "3": logic.right,
        }

        self.done = False

    def reset(self) -> None:
        self.matrix = logic.new_game(c.GRID_LEN)
        self.steps = 0
        self.score = 0
        self.rew = 0
        self.done = False

    def step(self) -> None:
        self.steps += 1
        if self.human and self.random:
            time.sleep(0.01)
        action = int(self.action) if not self.random else random.randint(0, 3)
        if self.human:
            print(f'Action: {["up", "down", "left", "right"][action]}')
        self.matrix, valid, self.rew = self.commands[str(action)](self.matrix)
        if valid:
            self.matrix = logic.add_two(self.matrix)
        else:
            self.rew -= 10
        if (state := logic.game_state(self.matrix)) == "win":
            self.rew += 10000
        elif state == "lose":
            self.rew -= 1000
        self.score += self.rew

    def get_observation(self) -> typing.List[float]:
        return [
            1 if cell == num else 0
            for num in [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
            for row in self.matrix
            for cell in row
        ]

    def get_reward(self) -> float:
        return self.rew

    def is_done(self) -> bool:
        return logic.game_state(self.matrix) in ["win", "lose"]

    def get_metrics(self) -> typing.List[float]:
        return [float(self.steps), float(self.score)]
