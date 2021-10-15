import random

import numpy as np

from ..config import SIMULATION_CONFIG
from ..models import Core, Direction, Node, Phase, Rail, Table
from ..simulation import Factory


def factory_from_config(config):
    if config is None:
        config = SIMULATION_CONFIG
    if config.get("layout") == "small":
        return get_small_default_factory(**config)
    elif config.get("layout") == "medium":
        return get_medium_default_factory(**config)
    elif config.get("layout") == "big":
        return get_default_factory(**config)
    else:
        raise ValueError("Choose from either 'small' or 'big'.")


def get_default_factory(
    random_seed=None,
    num_tables=8,
    num_cores=3,
    num_phases=1,
    max_num_steps=1000,
    with_rails=True,
    **kwargs,
) -> Factory:
    """
                    19--01
                        |
                    00--01--20
                        |
                    16--01      18--03
                    |               |
    02--14--15--15--16--17------17--03
        |                           |
        14--10--09--08--07          |
        |               |           |
        |               07--06--05--03--04---04
        |               |
        14--13--12--11--07
    """
    if random_seed:
        np.random.seed(random_seed)
        random.seed(random_seed)

    node_19 = Node("pt19", coordinates=(4, 0))
    node_00 = Node("pt00", coordinates=(4, 1))  # TR
    node_01_c = Node("pt01_c", coordinates=(5, 0))
    node_01_b = Node("pt01_b", coordinates=(5, 1))
    node_01_a = Node("pt01_a", coordinates=(5, 2))
    node_20 = Node("pt20", coordinates=(6, 1))  # TI
    node_16_b = Node("pt16_b", coordinates=(4, 2))
    node_16_a = Node("pt16_a", coordinates=(4, 3))
    node_02 = Node("pt02", coordinates=(0, 3))
    node_14_c = Node("pt14_c", coordinates=(1, 3))
    node_14_b = Node("pt14_b", coordinates=(1, 4))
    node_14_a = Node("pt14_a", coordinates=(1, 6))
    node_15_a = Node("pt15_a", coordinates=(2, 3))
    node_15_b = Node("pt15_b", coordinates=(3, 3))
    node_17_a = Node("pt17_a", coordinates=(5, 3))
    node_17_b = Node("pt17_b", coordinates=(7, 3))
    node_18 = Node("pt18", coordinates=(7, 2))
    node_03_c = Node("pt03_c", coordinates=(8, 2))
    node_03_b = Node("pt03_b", coordinates=(8, 3))
    node_03_a = Node("pt03_a", coordinates=(8, 5))
    node_04_a = Node("pt04_a", coordinates=(9, 5))
    node_04_b = Node("pt04_b", coordinates=(10, 5))
    node_05 = Node("pt05", coordinates=(7, 5))
    node_06 = Node("pt06", coordinates=(6, 5))
    node_07_c = Node("pt07_c", coordinates=(5, 4))
    node_07_b = Node("pt07_b", coordinates=(5, 5))
    node_07_a = Node("pt07_a", coordinates=(5, 6))
    node_08 = Node("pt08", coordinates=(4, 4))
    node_09 = Node("pt09", coordinates=(3, 4))
    node_10 = Node("pt10", coordinates=(2, 4))
    node_11 = Node("pt11", coordinates=(4, 6))
    node_12 = Node("pt12", coordinates=(3, 6))
    node_13 = Node("pt13", coordinates=(2, 6))

    node_19.add_neighbour(node_01_a, Direction.right)
    node_00.add_neighbour(node_01_b, Direction.right)
    node_20.add_neighbour(node_01_b, Direction.left)
    node_01_c.add_neighbour(node_01_b, Direction.down)
    node_01_b.add_neighbour(node_01_a, Direction.down)
    node_16_b.add_neighbour(node_01_c, Direction.right)
    node_16_b.add_neighbour(node_16_a, Direction.down)
    node_02.add_neighbour(node_14_c, Direction.right)
    node_14_c.add_neighbour(node_14_b, Direction.down)
    node_14_b.add_neighbour(node_14_a, Direction.down)
    node_14_c.add_neighbour(node_15_a, Direction.right)
    node_15_a.add_neighbour(node_15_b, Direction.right)
    node_15_b.add_neighbour(node_16_a, Direction.right)
    node_16_a.add_neighbour(node_17_a, Direction.right)
    node_17_a.add_neighbour(node_17_b, Direction.right)
    node_17_b.add_neighbour(node_03_b, Direction.right)
    node_18.add_neighbour(node_03_c, Direction.right)
    node_03_c.add_neighbour(node_03_b, Direction.down)
    node_03_b.add_neighbour(node_03_a, Direction.down)
    node_03_a.add_neighbour(node_04_a, Direction.right)
    node_04_a.add_neighbour(node_04_b, Direction.right)
    node_03_a.add_neighbour(node_05, Direction.left)
    node_05.add_neighbour(node_06, Direction.left, bidirectional=False)
    node_06.add_neighbour(node_07_b, Direction.left, bidirectional=False)
    node_07_c.add_neighbour(node_07_b, Direction.down)
    node_07_b.add_neighbour(node_07_a, Direction.down)
    node_07_c.add_neighbour(node_08, Direction.left, bidirectional=False)
    node_08.add_neighbour(node_09, Direction.left, bidirectional=False)
    node_09.add_neighbour(node_10, Direction.left, bidirectional=False)
    node_10.add_neighbour(node_14_b, Direction.left, bidirectional=False)
    node_07_a.add_neighbour(node_11, Direction.left, bidirectional=False)
    node_11.add_neighbour(node_12, Direction.left, bidirectional=False)
    node_12.add_neighbour(node_13, Direction.left, bidirectional=False)
    node_13.add_neighbour(node_14_a, Direction.left, bidirectional=False)

    nodes = [
        node_00,
        node_01_a,
        node_01_b,
        node_01_c,
        node_02,
        node_03_a,
        node_03_b,
        node_03_c,
        node_04_a,
        node_04_b,
        node_05,
        node_06,
        node_07_a,
        node_07_b,
        node_07_c,
        node_08,
        node_09,
        node_10,
        node_11,
        node_12,
        node_13,
        node_14_a,
        node_14_b,
        node_14_c,
        node_15_a,
        node_15_b,
        node_16_a,
        node_16_b,
        node_17_a,
        node_17_b,
        node_18,
        node_19,
        node_20,
    ]

    rails = []
    if with_rails:
        rail_01 = Rail(nodes=[node_01_a, node_01_b, node_01_c])
        rail_16 = Rail(nodes=[node_16_a, node_16_b])
        rail_14 = Rail(nodes=[node_14_a, node_14_b, node_14_c])
        rail_15 = Rail(nodes=[node_15_a, node_15_b])
        rail_17 = Rail(nodes=[node_17_a, node_17_b])
        rail_03 = Rail(nodes=[node_03_a, node_03_b, node_03_c])
        rail_04 = Rail(nodes=[node_04_a, node_04_b])
        rail_07 = Rail(nodes=[node_07_a, node_07_b, node_07_c])
        rails = [rail_01, rail_03, rail_04, rail_07, rail_14, rail_15, rail_16, rail_17]

    scenario = SIMULATION_CONFIG.get("scenario")

    if scenario == "fixed_2":
        table_nodes = [node_17_a, node_14_c]
        target_nodes = [node_00, node_06]
        tables = create_scenario(
            table_nodes, target_nodes, rails, num_tables, num_cores, num_phases
        )
    elif scenario == "fixed_4":
        table_nodes = [node_17_a, node_14_c, node_02, node_16_b]
        target_nodes = [node_00, node_06, node_04_a, node_00]
        tables = create_scenario(
            table_nodes, target_nodes, rails, num_tables, num_cores, num_phases
        )
    elif scenario == "fixed_6":
        table_nodes = [node_17_a, node_14_c, node_02, node_16_b, node_04_a, node_18]
        target_nodes = [node_00, node_06, node_06, node_04_a, node_00, node_04_a]
        tables = create_scenario(
            table_nodes, target_nodes, rails, num_tables, num_cores, num_phases
        )
    elif scenario == "fixed_8":
        table_nodes = [
            node_17_a,
            node_14_c,
            node_02,
            node_16_b,
            node_04_a,
            node_18,
            node_01_a,
            node_20,
        ]
        target_nodes = [
            node_00,
            node_06,
            node_06,
            node_04_a,
            node_00,
            node_04_a,
            node_04_a,
            node_06,
        ]
        tables = create_scenario(
            table_nodes, target_nodes, rails, num_tables, num_cores, num_phases
        )
    elif scenario == "fixed_10":
        table_nodes = [
            node_17_a,
            node_14_c,
            node_02,
            node_16_b,
            node_04_a,
            node_18,
            node_01_a,
            node_20,
            node_03_a,
            node_13,
        ]
        target_nodes = [
            node_00,
            node_06,
            node_06,
            node_04_a,
            node_00,
            node_04_a,
            node_04_a,
            node_06,
            node_14_a,
            node_14_a,
        ]
        tables = create_scenario(
            table_nodes, target_nodes, rails, num_tables, num_cores, num_phases
        )
    elif scenario == "fixed_12":
        table_nodes = [
            node_17_a,
            node_14_c,
            node_02,
            node_16_b,
            node_04_a,
            node_18,
            node_01_a,
            node_20,
            node_03_a,
            node_13,
            node_15_a,
            node_05,
        ]
        target_nodes = [
            node_00,
            node_06,
            node_06,
            node_04_a,
            node_00,
            node_04_a,
            node_04_a,
            node_06,
            node_14_a,
            node_14_a,
            node_00,
            node_14_a,
        ]
        tables = create_scenario(
            table_nodes, target_nodes, rails, num_tables, num_cores, num_phases
        )
    elif scenario == "random":
        tables = create_random_tables_and_cores(
            nodes, rails, num_tables, num_cores, num_phases
        )
    elif scenario == "random_fixed_targets":
        target_nodes = [node_00, node_06, node_06, node_04_a, node_00, node_04_a]
        tables = create_random_tables_fixed_target_nodes(
            nodes, target_nodes, rails, num_tables, num_cores, num_phases
        )
    else:
        raise ValueError(f"Unsupported scenario {scenario}.")

    return Factory(nodes, rails, tables, max_num_steps, "DefaultFactory")


def get_medium_default_factory(
    random_seed=None,
    num_tables=4,
    num_cores=2,
    num_phases=1,
    max_num_steps=1000,
    with_rails=True,
    **kwargs,
) -> Factory:
    """
    1--2-----2--3
    |           |
    1--6--5     |     9  10
    |     |     |     |  |
    |     5--4--3--8--8--8
    |     |           |
    1--7--5           11
    """
    if random_seed:
        random.seed(random_seed)
        np.random.seed(random_seed)

    node_1_c = Node("pt1_c", coordinates=(0, 0))
    node_1_b = Node("pt1_b", coordinates=(0, 1))
    node_1_a = Node("pt1_a", coordinates=(0, 3))
    node_2_a = Node("pt2_a", coordinates=(1, 0))
    node_2_b = Node("pt2_b", coordinates=(3, 0))
    node_3_b = Node("pt3_b", coordinates=(4, 0))
    node_3_a = Node("pt3_a", coordinates=(4, 2))
    node_4 = Node("pt4", coordinates=(3, 2))
    node_5_c = Node("pt5_c", coordinates=(2, 1))
    node_5_b = Node("pt5_b", coordinates=(2, 2))
    node_5_a = Node("pt5_a", coordinates=(2, 3))
    node_6 = Node("pt6", coordinates=(1, 1))
    node_7 = Node("pt7", coordinates=(1, 3))
    node_8_a = Node("pt8_a", coordinates=(5, 2))
    node_8_b = Node("pt8_b", coordinates=(6, 2))
    node_8_c = Node("pt8_c", coordinates=(7, 2))
    node_9 = Node("pt9", coordinates=(6, 1))
    node_10 = Node("pt10", coordinates=(7, 1))
    node_11 = Node("pt11", coordinates=(6, 3))

    node_1_c.add_neighbour(node_1_b, Direction.down)
    node_1_b.add_neighbour(node_1_a, Direction.down)
    node_1_c.add_neighbour(node_2_a, Direction.right)
    node_2_a.add_neighbour(node_2_b, Direction.right)
    node_2_b.add_neighbour(node_3_b, Direction.right)
    node_3_b.add_neighbour(node_3_a, Direction.down)
    node_3_a.add_neighbour(node_4, Direction.left)
    node_4.add_neighbour(node_5_b, Direction.left)
    node_5_c.add_neighbour(node_5_b, Direction.down)
    node_5_b.add_neighbour(node_5_a, Direction.down)
    node_5_c.add_neighbour(node_6, Direction.left)
    node_5_a.add_neighbour(node_7, Direction.left)
    node_6.add_neighbour(node_1_b, Direction.left)
    node_7.add_neighbour(node_1_a, Direction.left)
    node_3_a.add_neighbour(node_8_a, Direction.right)
    node_8_a.add_neighbour(node_8_b, Direction.right)
    node_8_b.add_neighbour(node_8_c, Direction.right)
    node_8_b.add_neighbour(node_9, Direction.up)
    node_8_b.add_neighbour(node_11, Direction.down)
    node_8_c.add_neighbour(node_10, Direction.up)

    nodes = [
        node_3_a,
        node_3_b,
        node_4,
        node_5_a,
        node_5_b,
        node_5_c,
        node_6,
        node_7,
        node_1_a,
        node_1_b,
        node_1_c,
        node_2_a,
        node_2_b,
        node_8_a,
        node_8_b,
        node_8_c,
        node_9,
        node_10,
        node_11,
    ]

    rails = []
    if with_rails:
        rail_1 = Rail(nodes=[node_1_a, node_1_b, node_1_c])
        rail_2 = Rail(nodes=[node_2_a, node_2_b])
        rail_3 = Rail(nodes=[node_3_a, node_3_b])
        rail_4 = Rail(nodes=[node_5_a, node_5_b, node_5_c])
        rail_8 = Rail(nodes=[node_8_a, node_8_b, node_8_c])
        rails = [rail_1, rail_2, rail_3, rail_4, rail_8]

    tables = create_random_tables_and_cores(
        nodes, rails, num_tables, num_cores, num_phases
    )

    return Factory(nodes, rails, tables, max_num_steps, "MediumDefaultFactory")


def get_small_default_factory(
    random_seed=None,
    num_tables=4,
    num_cores=2,
    num_phases=1,
    max_num_steps=1000,
    with_rails=True,
    **kwargs,
) -> Factory:
    """
    1--2-----2--3
    |           |
    1--6--5     |
    |     |     |
    |     5--4--3
    |     |
    1--7--5
    """
    if random_seed:
        random.seed(random_seed)
        np.random.seed(random_seed)

    node_1_c = Node("pt1_c", coordinates=(0, 0))
    node_1_b = Node("pt1_b", coordinates=(0, 1))
    node_1_a = Node("pt1_a", coordinates=(0, 3))
    node_2_a = Node("pt2_a", coordinates=(1, 0))
    node_2_b = Node("pt2_b", coordinates=(3, 0))
    node_3_b = Node("pt3_b", coordinates=(4, 0))
    node_3_a = Node("pt3_a", coordinates=(4, 2))
    node_4 = Node("pt4", coordinates=(3, 2))
    node_5_c = Node("pt5_c", coordinates=(2, 1))
    node_5_b = Node("pt5_b", coordinates=(2, 2))
    node_5_a = Node("pt5_a", coordinates=(2, 3))
    node_6 = Node("pt6", coordinates=(1, 1))
    node_7 = Node("pt7", coordinates=(1, 3))

    node_1_c.add_neighbour(node_1_b, Direction.down)
    node_1_b.add_neighbour(node_1_a, Direction.down)
    node_1_c.add_neighbour(node_2_a, Direction.right)
    node_2_a.add_neighbour(node_2_b, Direction.right)
    node_2_b.add_neighbour(node_3_b, Direction.right)
    node_3_b.add_neighbour(node_3_a, Direction.down)
    node_3_a.add_neighbour(node_4, Direction.left)
    node_4.add_neighbour(node_5_b, Direction.left)
    node_5_c.add_neighbour(node_5_b, Direction.down)
    node_5_b.add_neighbour(node_5_a, Direction.down)
    node_5_c.add_neighbour(node_6, Direction.left)
    node_5_a.add_neighbour(node_7, Direction.left)
    node_6.add_neighbour(node_1_b, Direction.left)
    node_7.add_neighbour(node_1_a, Direction.left)

    nodes = [
        node_3_a,
        node_3_b,
        node_4,
        node_5_a,
        node_5_b,
        node_5_c,
        node_6,
        node_7,
        node_1_a,
        node_1_b,
        node_1_c,
        node_2_a,
        node_2_b,
    ]

    rails = []
    if with_rails:
        rail_1 = Rail(nodes=[node_1_a, node_1_b, node_1_c])
        rail_2 = Rail(nodes=[node_2_a, node_2_b])
        rail_3 = Rail(nodes=[node_3_a, node_3_b])
        rail_4 = Rail(nodes=[node_5_a, node_5_b, node_5_c])
        rails = [rail_1, rail_2, rail_3, rail_4]

    tables = create_random_tables_and_cores(
        nodes, rails, num_tables, num_cores, num_phases
    )

    return Factory(nodes, rails, tables, max_num_steps, "SmallDefaultFactory")


def create_random_tables_and_cores(nodes, rails, num_tables, num_cores, num_phases):
    non_rail_nodes = [n for n in nodes if not n.is_rail]
    for rail in rails:
        # Add precisely one spot on each rail that a table could be placed on
        non_rail_nodes.append(random.choice(rail.nodes))

    tables = []
    # Randomly select a node for each table.
    table_indices = np.random.choice(
        range(len(non_rail_nodes)), num_tables, replace=False
    )
    for i in range(num_tables):
        tables.append(Table(non_rail_nodes[table_indices[i]], name=f"table_{i}"))

    # Core targets go on immobile nodes
    fixed_nodes = [n for n in nodes if not n.is_rail]
    for idx in range(num_cores):
        cycle = {}
        core_indices = np.random.choice(
            range(len(fixed_nodes)), num_phases, replace=False
        )
        for p in range(num_phases):
            # For each core phase, randomly select a fixed node.
            cycle[Phase(p)] = fixed_nodes[core_indices[p]]
        Core(tables[idx], cycle, f"core_{idx}")
    return tables


def create_scenario(
    table_nodes, target_nodes, rails, num_tables, num_cores, num_phases
):
    # TODO: check num_phases == 1,
    # check that table nodes are valid (no two tables exist on same rail),
    # check that target nodes are valid (no targets exist on rail)

    tables = []
    for i in range(num_tables):
        tables.append(Table(table_nodes[i], name=f"table_{i}"))

    for j in range(num_cores):
        cycle = {}
        cycle[Phase(0)] = target_nodes[j]
        Core(tables[j], cycle, f"core_{j}")
    return tables


def create_random_tables_fixed_target_nodes(
    nodes, target_nodes, rails, num_tables, num_cores, num_phases
):
    non_rail_nodes = [n for n in nodes if not n.is_rail]
    for rail in rails:
        # Add precisely one spot on each rail that a table could be placed on
        non_rail_nodes.append(random.choice(rail.nodes))

    tables = []
    # Randomly select a node for each table.
    table_indices = np.random.choice(
        range(len(non_rail_nodes)), num_tables, replace=False
    )
    for i in range(num_tables):
        tables.append(Table(non_rail_nodes[table_indices[i]], name=f"table_{i}"))

    # Core targets go on fixed nodes
    for idx in range(num_cores):
        cycle = {}
        core_indices = np.random.choice(
            range(len(target_nodes)), num_phases, replace=False
        )
        for p in range(num_phases):
            # For each core phase, randomly select a fixed target.
            cycle[Phase(p)] = target_nodes[core_indices[p]]
        Core(tables[idx], cycle, f"core_{idx}")
    return tables
