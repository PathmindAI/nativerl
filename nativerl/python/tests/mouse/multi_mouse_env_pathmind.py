import typing
from pathmind.simulation import Simulation, Discrete, Continuous


class MultiMouseAndCheese(Simulation):

    mouses = [(0, 0), (1, 1), (5, 5)]
    cheeses = [(4, 4), (3, 2), (0, 1)]
    moved = [False, False, False]
    steps = 0

    def number_of_agents(self) -> int:
        return 3

    def action_space(self, agent_id) -> typing.Union[Continuous, Discrete]:
        return Discrete(4)

    def reset(self) -> None:
        self.mouses = [(0, 0), (1, 1), (5, 5)]
        self.cheeses = [(4, 4), (3, 2), (0, 1)]
        self.steps = 0

    def step(self) -> None:
        self.steps += 1

        for i in range(self.number_of_agents()):
            if not self.is_done(i):
                self.moved[i] = True
                action = self.action[i][0]

                if action == 0:  # move up
                    self.mouses[i] = (min(self.mouses[i][0] + 1, 5), self.mouses[i][1])
                elif action == 1:  # move right
                    self.mouses[i] = (self.mouses[i][0], min(self.mouses[i][1] + 1, 5))
                elif action == 2:  # move down
                    self.mouses[i] = (max(self.mouses[i][0] - 1, 0), self.mouses[i][1])
                elif action == 3:  # move left
                    self.mouses[i] = (self.mouses[i][0], max(self.mouses[i][1] - 1, 0))
                else:
                    raise ValueError("Invalid action")
            else:
                self.moved[i] = False

    def get_observation(self, agent_id) -> typing.Dict[str, float]:
        return {
            "mouse_row":  float(self.mouses[agent_id][0]) / 5.0,
            "mouse_col": float(self.mouses[agent_id][1]) / 5.0,
            "mouse_row_dist": abs(self.cheeses[agent_id][0] - self.mouses[agent_id][0]) / 5.0,
            "mouse_col_dist": abs(self.cheeses[agent_id][1] - self.mouses[agent_id][1]) / 5.0,
        }

    def get_reward(self, agent_id) -> typing.Dict[str, float]:
        return {
            "found_cheese": 1 if self.is_done(agent_id) and self.moved[agent_id] else 0
        }

    def is_done(self, agent_id) -> bool:
        return self.mouses[agent_id] == self.cheeses[agent_id]



