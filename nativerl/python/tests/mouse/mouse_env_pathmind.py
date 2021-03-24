import typing
from pathmind.simulation import Simulation, Discrete, Continuous


class MouseAndCheese(Simulation):

    mouse = (0, 0)
    cheese = (4, 4)
    steps = 0

    def number_of_agents(self) -> int:
        return 1

    def action_space(self, agent_id) -> typing.Union[Continuous, Discrete]:
        return Discrete(4)

    def reset(self) -> None:
        self.mouse = (0, 0)
        self.cheese = (4, 4)
        self.steps = 0

    def step(self) -> None:
        self.steps += 1

        action = self.action[0]

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

    def get_observation(self, agent_id) -> typing.Dict[str, float]:
        return {
            "mouse_row":  float(self.mouse[0]) / 5.0,
            "mouse_col": float(self.mouse[1]) / 5.0,
            "mouse_row_dist": abs(self.cheese[0] - self.mouse[0]) / 5.0,
            "mouse_col_dist": abs(self.cheese[1] - self.mouse[1]) / 5.0,
        }

    def get_reward(self, agent_id) -> typing.Dict[str, float]:
        return {
            "found_cheese": 1 if self.mouse == self.cheese else 0
        }

    def is_done(self, agent_id) -> bool:
        return self.mouse == self.cheese



