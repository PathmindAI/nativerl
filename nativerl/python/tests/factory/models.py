"""Models specify the basic building blocks for this project (what).
They model what objects are and how to interact with them."""
import enum
from typing import Optional, Tuple, TypeVar, List, Dict


__all__ = ["Direction", "Node", "Rail", "Table", "Phase", "Core", "ActionResult"]
N = TypeVar('N', bound='Node')


class Direction(enum.IntEnum):
    """Four basic directions on a 2D plane."""
    up = 0
    right = 1
    down = 2
    left = 3

    @staticmethod
    def get_all():
        return [d.name for d in Direction]

    def opposite(self):
        return Direction((self.value + 2) % 4)


class Table:
    """Tables sit on nodes and potentially carry cores. Tables will
    act as our main agents in simulations.
    """
    def __init__(self, node, core=None, name: str = None):
        node.set_table(self)
        self.node = node
        self.core = core
        self.name = name
        # Since tables act as agents, they (not cores) know when they're at a target
        self.is_at_target = False
    
    def has_core(self) -> bool:
        return self.core is not None

    def set_core(self, core) -> None:
        self.core = core
    
    def set_node(self, node) -> None:
        self.node = node
    
    def get_target(self):
        return self.core.current_target if self.has_core() else None
    
    def phase_completed(self):
        """Complete a phase on behalf of your core. If all phases are complete,
        remove the core from this table."""
        if self.has_core():
            self.core.phase_completed()
            if self.core.done():
                self.core = None


class Node:
    """Basic building block for this problem. Nodes can be connected
    to each other. A node can either belong to a Rail, or exist as
    a standalone, static node. Nodes can host a single Table.
    """
    def __init__(self, name: Optional[str] = None, is_rail: bool = False,
                 coordinates: Optional[Tuple[int, int]] = None):
        self.is_rail = is_rail
        self.name = name
        self.coordinates = coordinates
        self.neighbours = {d: None for d in Direction.get_all()}
        self.table: Optional[Table] = None

    def has_table(self) -> bool:
        return self.table is not None

    def set_table(self, table: Table) -> None:
        assert not self.has_table(), "Can't set another Table, remove the existing first."
        self.table = table

    def remove_table(self) -> None:
        self.table = None
    
    def connected_to(self, node: N) -> bool:
        return node in [n for n in self.neighbours.values()]

    def get_neighbour(self, where: Direction) -> N:
        return self.neighbours[where.name]

    def has_neighbour(self, where: Direction) -> bool:
        return self.get_neighbour(where) is not None
    
    def add_neighbour(self, neighbour: N, where: Direction, bidirectional=True) -> None:
        assert neighbour is not self, "Can't connect node to itself"
        opposite = where.opposite()
        assert not self.has_neighbour(where) and not neighbour.has_neighbour(opposite)
        self.neighbours[where.name] = neighbour
        if bidirectional:
            neighbour.neighbours[opposite.name] = self


class Rail:
    """Rails consist of sequentially connected Nodes,
    only one of which can carry a Table."""
    def __init__(self, nodes: List[Node]):
        for node in nodes:
            node.is_rail = True
        for i in range(len(nodes)-1):
            assert nodes[i].connected_to(nodes[i+1])
        self.nodes = nodes

    def get_table_node(self):
        num_tables = self.num_tables()
        assert num_tables <= 1, "Can have at most one table on a rail"
        if num_tables == 0:
            return None
        return [n for n in self.nodes if n.has_table()][0]

    def num_tables(self):
        return len([n for n in self.nodes if n.has_table()])

    def has_table(self):
        return self.num_tables() == 1
    
    def is_free(self) -> bool:
        return self.num_tables() == 0



class Phase(enum.IntEnum):
    """Cores go through one or several production phases.
    Each phase is mapped to a Node"""
    A = 0
    B = 1
    C = 2
    D = 3
    E = 4
    F = 5


class Core:
    """Cores reside on tables and know which phase corresponds to which
    target node. Cores have to be delivered to targets according to the
    specified phases."""
    def __init__(self, table: Table, cycle: Dict[Phase, Node], name: str = None):
        self.table = table
        table.set_core(self)
        self.cycle = cycle
        self.num_phases = len(cycle)
        self.current_phase: Phase = list(cycle)[0]
        self.current_target: Node = cycle[self.current_phase]
        self.name = name

    def done(self):
        return not bool(self.cycle)
    
    def phase_completed(self):
        """Remove the completed target from the phases and set next target."""
        del self.cycle[self.current_phase]
        self.current_phase = list(self.cycle)[0] if not self.done() else None
        self.current_target = self.cycle[self.current_phase] if not self.done() else None


class ActionResult(enum.IntEnum):
    """Result of an action with attached rewards."""
    NONE = 0,
    MOVED = 1,
    INVALID = 2
    COLLISION = 3
    INVALID_RAIL_ENTERING = 4

    def reward(self):
        return 0 if self.value < 2 else -1
