""""Controls specify how objects change state (how)."""
import enum
import random

from .models import ActionResult, Direction, Node, Rail, Table
from .simulation import Factory


class Action(enum.IntEnum):
    """Move in a direction or stay where you are."""

    up = 0
    right = 1
    down = 2
    left = 3
    none = 4

    @staticmethod
    def random_action():
        return Action(random.randrange(0, 5))


def do_action(table: Table, factory: Factory, action: Action):
    return TableAndRailController(factory).do_action(table, action)


class TableAndRailController:
    def __init__(self, factory: Factory, name=None):
        self.factory = factory
        self.name = name

    @staticmethod
    def _move_table(table: Table, to: Node) -> ActionResult:
        """Move table to an adjacent node. Cores are moved automatically.
        If we move on a rail, also move the shuttle. If the destination
        completes a phase, mark it as such.
        """
        start = table.node

        # Remove table from "start" node
        start.remove_table()

        # Put table on "to" node
        table.set_node(to)
        to.set_table(table)

        if table.get_target() is to:
            table.phase_completed()
            table.is_at_target = True
        else:
            table.is_at_target = False

        return ActionResult.MOVED

    def _move_to_rail(self, table: Table, rail: Rail, neighbour: Node) -> ActionResult:
        raise NotImplementedError

    def do_action(self, table: Table, action: Action) -> ActionResult:
        """Attempt to carry out a specified action."""
        table.is_at_target = False  # Reset target
        if action.value == 4:
            return ActionResult.NONE
        direction = Direction(action.value)
        has_neighbour = table.node.has_neighbour(direction)
        if not has_neighbour:
            return ActionResult.INVALID
        else:
            neighbour = table.node.get_neighbour(direction)
            if neighbour.has_table():
                return ActionResult.COLLISION
            if neighbour.is_rail:  # node -> rail or rail -> rail
                # can we hop on the rail?
                rail = self.factory.get_rail(node=neighbour)
                assert rail.num_tables() <= 1, "A rail can have at most one table"
                if rail.is_free() or table.node in rail.nodes:
                    return self._move_table(table, neighbour)
                else:
                    # target is blocked with a table.
                    return ActionResult.INVALID_RAIL_ENTERING
            else:  # Move table from a) node -> node or b) rail -> node
                return self._move_table(table, neighbour)
