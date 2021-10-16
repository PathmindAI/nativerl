import typing

from gym import Env, spaces


class MouseAndCheese(Env):

    mouse = (0, 0)
    cheese = (4, 4)
    number_of_actions = 4
    number_of_observations = 4
    steps = 0

    def __init__(self):
        self.action_space = spaces.Discrete(self.number_of_actions)
        self.observation_space = spaces.Box(0, 1, (self.number_of_observations,))

    def reset(self):
        self.mouse = (0, 0)
        self.cheese = (4, 4)
        self.steps = 0

        return self.get_observation()

    def step(self, action):
        self.steps += 1

        if action == 0:  # move up
            self.mouse = (min(self.mouse[0] + 1, 5), self.mouse[1])
        elif action == 1:  # move right
            self.mouse = (self.mouse[0], min(self.mouse[1] + 1, 5))
        elif action == 2:  # move down
            self.mouse = (max(self.mouse[0] - 1, 0), self.mouse[1])
        elif action == 3:  # move left
            self.mouse = (self.mouse[0], max(self.mouse[1] - 1, 0))
        else:
            raise ValueError("Invalid action")

        print(self.mouse)

        return self.get_observation(), self.get_reward(), self.is_done(), {}

    def get_observation(self) -> typing.List[float]:
        return [
            float(self.mouse[0]) / 5.0,
            float(self.mouse[1]) / 5.0,
            abs(self.cheese[0] - self.mouse[0]) / 5.0,
            abs(self.cheese[1] - self.mouse[1]) / 5.0,
        ]

    def get_reward(self) -> float:
        return 1 if self.mouse == self.cheese else 0

    def is_done(self) -> bool:
        return self.mouse == self.cheese
