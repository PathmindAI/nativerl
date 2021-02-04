from typing import Optional, List, Dict, TypeVar
import os
from collections import Counter
from csv import writer
import pprint
from .models import *

PRINTER = pprint.PrettyPrinter(indent=2)
VERBOSE = True

F = TypeVar('F', bound='Factory')


class Factory:
    """A Factory sets up all components (nodes, rails, tables) needed to
    solve the problem of delivering cores to their destinations. Note that
    this is just a "model", the application logic and agent interaction is
    separated  """
    def __init__(self, nodes: List[Node], rails: List[Rail],
                 tables: List[Table], max_num_steps: int  = 1000, name: str = None):
        self.nodes = nodes
        self.rails = rails
        self.tables = tables
        self.name = name
        self.cores = [t.core for t in self.tables if t.has_core()]
        self.max_num_steps = max_num_steps
        self.initial_max_num_steps = max_num_steps

        # Stats counter
        self.step_count = 0
        self.agent_step_counter: Dict[int, int] = {t: 0 for t in range(len(self.tables))}
        self.moves: Dict[int, List[ActionResult]] = {t: [] for t in range(len(self.tables))}
        self.move_counter = Counter()
        self.action_counter = Counter()
        self.step_completion_counter: Dict[int, List[int]] = {t: [] for t in range(len(self.tables))}

    def is_solved(self):
        """A factory is solved if no table has a core anymore."""
        return len([t for t in self.tables if t.has_core()]) == 0

    def done(self):
        return all([c.done() for c in self.cores])

    def set_tables(self, tables: List[Table]):
        self.tables = tables

    def get_rail(self, node: Node) -> Optional[Rail]:
        for rail in self.rails:
            if node in rail.nodes:
                return rail
        return None

    def add_move(self, agent_id: int, action, move: ActionResult):
        self.step_count += 1
        self.moves.get(agent_id).append(move)
        self.agent_step_counter[agent_id] += 1
        self.move_counter[move.name] += 1
        self.action_counter[action.name] +=1

    def add_completed_step_count(self):
        for agent_id in range(len(self.tables)):
            counter = self.step_completion_counter.get(agent_id)
            counter.append(self.agent_step_counter[agent_id])

    def print_stats(self, episodes=None):
        """Print statistics to stdout for quick sanity checks."""
        if VERBOSE:
            PRINTER.pprint(">>> Completed an episode")
            PRINTER.pprint("   >>> Number of episodes completed:")
            PRINTER.pprint(episodes)
            PRINTER.pprint("   >>> Number of cores left to deliver:")
            cores_left = len([t for t in self.tables if t.has_core()])
            PRINTER.pprint(cores_left)
            PRINTER.pprint("   >>> Move counter")
            PRINTER.pprint(dict(self.move_counter))
            PRINTER.pprint("   >>> Action counter")
            PRINTER.pprint(dict(self.action_counter))
            PRINTER.pprint("   >>> Steps taken to completion")
            PRINTER.pprint(self.step_completion_counter)

    def record_stats(self):
        """Record statistics in a CSV file for later visualisation."""
        move_dict = dict(self.move_counter)
        move_dict['CORES_REMAIN'] = len([t for t in self.tables if t.has_core()])
        key_list = sorted(move_dict)
        elements = []
        for item in key_list:
            elements.append(move_dict[item])
        if os.path.exists(os.path.join(os.path.abspath('PPO/'), 'Move_Stats.csv')):
            with open(os.path.join(os.path.abspath('PPO/'), 'Move_Stats.csv'), 'a', newline='') as f:
                writer(f).writerow(elements)
        else:
            with open(os.path.join(os.path.abspath('PPO/'), 'Move_Stats.csv'), 'w+', newline='') as f:
                writer(f).writerow(key_list)
                writer(f).writerow(elements)
