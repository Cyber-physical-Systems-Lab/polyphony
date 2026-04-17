import numpy as np

from tarware.definitions import CollisionLayers
from tarware.warehouse import Action, Direction, RewardType, Warehouse


def make_env(allow_busy_replan: bool, find_path_agent_aware_always: bool = False) -> Warehouse:
    env = Warehouse(
        shelf_columns=3,
        column_height=8,
        shelf_rows=1,
        num_agvs=1,
        num_pickers=1,
        request_queue_size=5,
        max_inactivity_steps=200,
        max_steps=5000,
        reward_type=RewardType.INDIVIDUAL,
        observation_type="global",
        allow_busy_replan=allow_busy_replan,
        find_path_agent_aware_always=find_path_agent_aware_always,
    )
    env.reset(seed=0)
    return env


def make_multi_env(
    allow_busy_replan: bool,
    num_agvs: int,
    num_pickers: int,
    find_path_agent_aware_always: bool = False,
) -> Warehouse:
    env = Warehouse(
        shelf_columns=3,
        column_height=8,
        shelf_rows=1,
        num_agvs=num_agvs,
        num_pickers=num_pickers,
        request_queue_size=5,
        max_inactivity_steps=200,
        max_steps=5000,
        reward_type=RewardType.INDIVIDUAL,
        observation_type="global",
        allow_busy_replan=allow_busy_replan,
        find_path_agent_aware_always=find_path_agent_aware_always,
    )
    env.reset(seed=0)
    return env


def reachable_actions(env: Warehouse, agent_idx: int, min_path_len: int = 1) -> list[int]:
    agent = env.agents[agent_idx]
    actions: list[int] = []
    for action_id, coords in env.action_id_to_coords_map.items():
        path = env.find_path((agent.y, agent.x), coords, agent, care_for_agents=False)
        if len(path) >= min_path_len:
            actions.append(int(action_id))
    return actions


def shelf_action_at(env: Warehouse, x: int, y: int) -> int:
    for action_id in env.shelf_action_ids:
        coords = env.action_id_to_coords_map[action_id]
        if (int(coords[1]), int(coords[0])) == (x, y):
            return int(action_id)
    raise AssertionError(f"No shelf action found at {(x, y)}")


def charging_action_ids(env: Warehouse) -> list[int]:
    start = len(env.action_id_to_coords_map) - len(env.charging_stations) + 1
    return list(range(start, len(env.action_id_to_coords_map) + 1))


def test_legacy_busy_agent_ignores_new_macro_action() -> None:
    env = make_env(allow_busy_replan=False)
    try:
        first_action, second_action = reachable_actions(env, 0, min_path_len=2)[:2]
        env.step([first_action, 0])
        assert env.agents[0].busy
        assert env.agents[0].target == first_action

        env.step([second_action, 0])
        assert env.agents[0].target == first_action
    finally:
        env.close()


def test_opt_in_busy_agv_accepts_new_target_immediately() -> None:
    env = make_env(allow_busy_replan=True)
    try:
        first_action, second_action = reachable_actions(env, 0, min_path_len=2)[:2]
        env.step([first_action, 0])
        assert env.agents[0].busy
        env.step([second_action, 0])
        assert env.agents[0].target == second_action
    finally:
        env.close()


def test_opt_in_busy_picker_accepts_new_target_immediately() -> None:
    env = make_env(allow_busy_replan=True)
    try:
        first_action, second_action = reachable_actions(env, 1, min_path_len=2)[:2]
        env.step([0, first_action])
        assert env.agents[1].busy
        env.step([0, second_action])
        assert env.agents[1].target == second_action
    finally:
        env.close()


def test_opt_in_noop_cancels_busy_assignment() -> None:
    env = make_env(allow_busy_replan=True)
    try:
        first_action = reachable_actions(env, 0, min_path_len=2)[0]
        env.step([first_action, 0])
        assert env.agents[0].busy

        env.step([0, 0])
        assert not env.agents[0].busy
        assert env.agents[0].target == 0
        assert env.agents[0].path == []
        assert env.agents[0].req_action == Action.NOOP
    finally:
        env.close()


def test_opt_in_busy_charging_agent_can_be_redirected() -> None:
    env = make_env(allow_busy_replan=True)
    try:
        agv = env.agents[0]
        station = env.charging_stations[0]
        agv.x = int(station.x)
        agv.y = int(station.y)
        agv.battery = 10
        charge_action = charging_action_ids(env)[0]
        agv.target = charge_action
        agv.path = []
        agv.busy = True
        agv.charging = True
        env._recalc_grid()

        redirect_action = reachable_actions(env, 0, min_path_len=1)[0]
        env.step([redirect_action, 0])
        assert env.agents[0].target == redirect_action
        assert not env.agents[0].charging
        assert env.agents[0].req_action != Action.CHARGE
    finally:
        env.close()


def test_find_path_agent_aware_always_overrides_care_for_agents_false() -> None:
    env = make_env(allow_busy_replan=True, find_path_agent_aware_always=True)
    try:
        agv = env.agents[0]
        picker = env.agents[1]
        agv.x = 0
        agv.y = 0
        picker.x = 1
        picker.y = 0
        env._recalc_grid()

        path = env.find_path((agv.y, agv.x), (0, 2), agv, care_for_agents=False)
        assert path
        assert (0, 1) not in path
    finally:
        env.close()


def test_find_path_agent_aware_always_falls_back_when_override_blocks_only_route() -> None:
    env = make_env(allow_busy_replan=True, find_path_agent_aware_always=True)
    try:
        agv = env.agents[0]
        picker = env.agents[1]
        agv.x = 0
        agv.y = 0
        picker.x = 1
        picker.y = 0
        env._recalc_grid()

        target_action = charging_action_ids(env)[1]
        target_coords = env.action_id_to_coords_map[target_action]

        explicit_agent_aware_path = env.find_path((agv.y, agv.x), target_coords, agv, care_for_agents=True)
        fallback_path = env.find_path((agv.y, agv.x), target_coords, agv, care_for_agents=False)

        assert explicit_agent_aware_path == []
        assert fallback_path
        assert fallback_path[0] == (1, 0)

        env._assign_macro_target(agv, target_action, {(station.x, station.y) for station in env.charging_stations})

        assert agv.target == target_action
        assert agv.busy
        assert agv.path
        assert agv.path[0] == (1, 0)
    finally:
        env.close()


def test_find_path_agent_aware_override_fallback_not_used_when_explicit_agent_aware_requested() -> None:
    env = make_env(allow_busy_replan=True, find_path_agent_aware_always=True)
    try:
        agv = env.agents[0]
        picker = env.agents[1]
        agv.x = 0
        agv.y = 0
        picker.x = 1
        picker.y = 0
        env._recalc_grid()

        target_coords = env.action_id_to_coords_map[charging_action_ids(env)[1]]
        path = env.find_path((agv.y, agv.x), target_coords, agv, care_for_agents=True)
        assert path == []
    finally:
        env.close()


def test_find_path_agent_aware_override_keeps_no_path_when_both_searches_fail() -> None:
    env = make_env(allow_busy_replan=True, find_path_agent_aware_always=True)
    try:
        agv = env.agents[0]
        picker = env.agents[1]
        agv.x = 0
        agv.y = 0
        picker.x = 1
        picker.y = 0
        env.grid[CollisionLayers.SHELVES, 0, 2] = 1
        env._recalc_grid()

        target_coords = (0, 3)
        path = env.find_path((agv.y, agv.x), target_coords, agv, care_for_agents=False)
        assert path == []
    finally:
        env.close()


def test_picker_can_take_final_step_onto_charging_station() -> None:
    env = make_env(allow_busy_replan=True)
    try:
        picker = env.agents[1]
        charge_action = charging_action_ids(env)[0]
        charge_coords = env.action_id_to_coords_map[charge_action]
        charge_x, charge_y = int(charge_coords[1]), int(charge_coords[0])

        picker.x = charge_x + 1
        picker.y = charge_y
        picker.battery = 0
        picker.busy = False
        picker.charging = False
        picker.target = 0
        picker.path = []
        env._recalc_grid()

        env.step([0, charge_action])

        assert env.agents[1].target == charge_action
        assert env.agents[1].req_action == Action.FORWARD
        assert (env.agents[1].x, env.agents[1].y) == (charge_x, charge_y)
    finally:
        env.close()


def test_picker_stops_two_steps_before_shelf_when_agv_is_not_waiting_there() -> None:
    env = make_env(allow_busy_replan=True)
    try:
        picker = env.agents[1]
        target_action = shelf_action_at(env, 6, 5)
        target_coords = env.action_id_to_coords_map[target_action]
        target_x, target_y = int(target_coords[1]), int(target_coords[0])

        picker.x = target_x
        picker.y = target_y - 2
        picker.target = 0
        picker.busy = False
        picker.charging = False
        picker.req_action = Action.NOOP
        picker.path = []

        env.agents[0].x = target_x - 2
        env.agents[0].y = target_y
        env.agents[0].target = 0
        env.agents[0].busy = False
        env.agents[0].req_action = Action.NOOP
        env._recalc_grid()

        env.step([0, target_action])

        assert env.agents[1].target == target_action
        assert env.agents[1].req_action == Action.NOOP
        assert (env.agents[1].x, env.agents[1].y) == (target_x, target_y - 2)
    finally:
        env.close()


def test_picker_can_resume_from_two_steps_out_when_agv_is_waiting_on_shelf() -> None:
    env = make_env(allow_busy_replan=True)
    try:
        agv = env.agents[0]
        picker = env.agents[1]
        target_action = shelf_action_at(env, 6, 5)
        target_coords = env.action_id_to_coords_map[target_action]
        target_x, target_y = int(target_coords[1]), int(target_coords[0])

        agv.x = target_x
        agv.y = target_y
        agv.target = target_action
        agv.busy = True
        agv.path = []
        agv.req_action = Action.TOGGLE_LOAD
        agv.carrying_shelf = None

        picker.x = target_x
        picker.y = target_y - 2
        picker.target = target_action
        picker.busy = True
        picker.charging = False
        picker.req_action = Action.NOOP
        picker.path = env.find_path((picker.y, picker.x), target_coords, picker, care_for_agents=False)
        env._recalc_grid()

        env.step([0, target_action])

        assert env.agents[1].target == target_action
        assert env.agents[1].req_action != Action.NOOP
        assert (env.agents[1].x, env.agents[1].y) != (target_x, target_y - 2)
    finally:
        env.close()


def test_picker_same_target_reassignment_rearms_final_support_step_when_idle() -> None:
    env = make_env(allow_busy_replan=True)
    try:
        agv = env.agents[0]
        picker = env.agents[1]
        target_action = shelf_action_at(env, 6, 5)
        target_coords = env.action_id_to_coords_map[target_action]
        target_x, target_y = int(target_coords[1]), int(target_coords[0])

        agv.x = target_x
        agv.y = target_y
        agv.target = target_action
        agv.busy = True
        agv.path = []
        agv.req_action = Action.TOGGLE_LOAD
        agv.carrying_shelf = None

        picker.x = target_x
        picker.y = target_y - 1
        picker.target = target_action
        picker.busy = False
        picker.charging = False
        picker.req_action = Action.NOOP
        picker.path = env.find_path((picker.y, picker.x), target_coords, picker, care_for_agents=False)

        env._recalc_grid()
        env.step([0, target_action])

        assert env.agents[1].target == target_action
        assert env.agents[1].req_action == Action.FORWARD
        assert (env.agents[1].x, env.agents[1].y) == (target_x, target_y)
    finally:
        env.close()


def test_same_step_picker_arrival_allows_unload() -> None:
    env = make_env(allow_busy_replan=True)
    try:
        delivered_shelf = env.shelfs[0]
        agv = env.agents[0]
        picker = env.agents[1]
        target_action = shelf_action_at(env, int(delivered_shelf.x), int(delivered_shelf.y))
        target_x = int(delivered_shelf.x)
        target_y = int(delivered_shelf.y)

        agv.x = target_x
        agv.y = target_y
        agv.target = target_action
        agv.busy = True
        agv.req_action = Action.TOGGLE_LOAD
        agv.carrying_shelf = delivered_shelf
        agv.has_delivered = True

        if target_y + 1 < env.grid_size[0]:
            picker.x = target_x
            picker.y = target_y + 1
            picker.dir = Direction.UP
        else:
            picker.x = target_x
            picker.y = target_y - 1
            picker.dir = Direction.DOWN
        picker.target = target_action
        picker.busy = True
        picker.req_action = Action.FORWARD
        picker.path = [(target_y, target_x)]

        env._recalc_grid()
        rewards = env.execute_micro_actions(np.zeros(env.num_agents))

        assert (picker.x, picker.y) == (target_x, target_y)
        assert agv.carrying_shelf is None
        assert env.grid[CollisionLayers.SHELVES, target_y, target_x] == delivered_shelf.id
        assert rewards[1] >= 0.0
    finally:
        env.close()


def test_unload_still_requires_picker_on_same_cell() -> None:
    env = make_env(allow_busy_replan=True)
    try:
        delivered_shelf = env.shelfs[0]
        agv = env.agents[0]
        picker = env.agents[1]
        target_x = int(delivered_shelf.x)
        target_y = int(delivered_shelf.y)

        agv.x = target_x
        agv.y = target_y
        agv.req_action = Action.TOGGLE_LOAD
        agv.target = shelf_action_at(env, target_x, target_y)
        agv.busy = True
        agv.carrying_shelf = delivered_shelf
        agv.has_delivered = True

        picker.x = target_x
        picker.y = max(0, target_y - 1)
        picker.req_action = Action.NOOP
        picker.busy = True
        picker.target = agv.target

        env._recalc_grid()
        env.execute_micro_actions(np.zeros(env.num_agents))

        assert agv.carrying_shelf is delivered_shelf
        assert env.grid[CollisionLayers.SHELVES, target_y, target_x] == 0
    finally:
        env.close()


def test_requested_shelf_claim_keeps_closer_agv_and_clears_farther_agv() -> None:
    env = make_multi_env(allow_busy_replan=True, num_agvs=2, num_pickers=1)
    try:
        requested_shelf = env.request_queue[0]
        target_action = shelf_action_at(env, int(requested_shelf.x), int(requested_shelf.y))
        near_agv = env.agents[0]
        far_agv = env.agents[1]

        near_agv.x = int(requested_shelf.x) + 1
        near_agv.y = int(requested_shelf.y)
        far_agv.x = 0
        far_agv.y = 0
        env._recalc_grid()

        env.attribute_macro_actions([target_action, target_action, 0])
        owners = env._resolve_agv_shelf_claims()

        assert owners[target_action] == near_agv.id
        assert near_agv.target == target_action
        assert near_agv.busy is True
        assert far_agv.target == 0
        assert far_agv.busy is False
        assert far_agv.req_action == Action.NOOP
        assert far_agv.unreachable_target_action_id == target_action
        assert far_agv.unreachable_target_reason == "claimed_by_other_agv"
    finally:
        env.close()


def test_requested_shelf_claim_prefers_agv_already_at_shelf() -> None:
    env = make_multi_env(allow_busy_replan=True, num_agvs=2, num_pickers=1)
    try:
        requested_shelf = env.request_queue[0]
        target_action = shelf_action_at(env, int(requested_shelf.x), int(requested_shelf.y))
        owner_agv = env.agents[0]
        other_agv = env.agents[1]

        owner_agv.x = int(requested_shelf.x)
        owner_agv.y = int(requested_shelf.y)
        owner_agv.target = target_action
        owner_agv.busy = True
        owner_agv.path = []
        owner_agv.req_action = Action.TOGGLE_LOAD

        other_agv.x = int(requested_shelf.x) + 1
        other_agv.y = int(requested_shelf.y)
        env._recalc_grid()

        env.attribute_macro_actions([target_action, target_action, 0])
        owners = env._resolve_agv_shelf_claims()

        assert owners[target_action] == owner_agv.id
        assert owner_agv.target == target_action
        assert owner_agv.req_action == Action.TOGGLE_LOAD
        assert other_agv.target == 0
        assert other_agv.unreachable_target_reason == "claimed_by_other_agv"
    finally:
        env.close()


def test_requested_shelf_claim_tie_breaks_by_lower_agv_id() -> None:
    env = make_multi_env(allow_busy_replan=True, num_agvs=2, num_pickers=1)
    try:
        requested_shelf = env.request_queue[0]
        target_action = shelf_action_at(env, int(requested_shelf.x), int(requested_shelf.y))
        agv_0 = env.agents[0]
        agv_1 = env.agents[1]

        agv_0.x = int(requested_shelf.x) + 1
        agv_0.y = int(requested_shelf.y)
        agv_1.x = int(requested_shelf.x) - 1
        agv_1.y = int(requested_shelf.y)
        env._recalc_grid()

        env.attribute_macro_actions([target_action, target_action, 0])
        owners = env._resolve_agv_shelf_claims()

        assert owners[target_action] == agv_0.id
        assert agv_0.target == target_action
        assert agv_1.target == 0
        assert agv_1.unreachable_target_reason == "claimed_by_other_agv"
    finally:
        env.close()


def test_opt_in_interrupts_terminal_load_before_toggle() -> None:
    env = make_env(allow_busy_replan=True)
    try:
        agv = env.agents[0]
        picker = env.agents[1]
        target_action = env.shelf_action_ids[0]
        target_coords = env.action_id_to_coords_map[target_action]
        target_x, target_y = int(target_coords[1]), int(target_coords[0])

        agv.x = target_x
        agv.y = target_y
        picker.x = target_x
        picker.y = target_y
        agv.busy = True
        agv.path = []
        agv.target = target_action
        agv.charging = False
        agv.carrying_shelf = None
        env._recalc_grid()

        shelf_id = int(env.grid[CollisionLayers.SHELVES, target_y, target_x])
        shelf = env.shelfs[shelf_id - 1]
        env.request_queue = [shelf]

        redirect_action = next(
            action_id
            for action_id in reachable_actions(env, 0, min_path_len=1)
            if action_id != target_action
        )
        env.step([redirect_action, 0])
        assert env.agents[0].target == redirect_action
        assert env.agents[0].carrying_shelf is None
        assert env.agents[0].req_action != Action.TOGGLE_LOAD
    finally:
        env.close()
