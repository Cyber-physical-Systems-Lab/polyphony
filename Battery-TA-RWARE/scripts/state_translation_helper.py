from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# This module is the state-to-language translation layer.
# Lower-level helpers expose action ids, distances, and state facts.
# Higher-level helpers turn those facts into natural-language strings.
# Those word-level renderings are intentionally isolated here because they will
# likely evolve during the JSON-vs-natural-language prompt study.


# ---------------------------------------------------------------------------
# Action-space and environment lookup helpers
# ---------------------------------------------------------------------------
def classify_action(env, action_id: int) -> str:
    """Map an environment action id to a coarse action class."""
    if action_id == 0:
        return "NOOP"
    if 1 <= action_id <= len(env.goals):
        return "GOAL"
    charging_start = len(env.action_id_to_coords_map) - len(env.charging_stations) + 1
    if charging_start <= action_id <= len(env.action_id_to_coords_map):
        return "CHARGING"
    return "SHELF"


def action_coords(env, action_id: int) -> Optional[Tuple[int, int]]:
    """Return the internal `(y, x)` grid coordinate for an action id."""
    return env.action_id_to_coords_map.get(int(action_id))


def goal_action_ids(env) -> List[int]:
    """Return the contiguous action-id span used for goal cells."""
    return list(range(1, len(env.goals) + 1))


def charging_action_ids(env) -> List[int]:
    """Return all action ids that correspond to charging stations."""
    start = len(env.action_id_to_coords_map) - len(env.charging_stations) + 1
    return list(range(start, len(env.action_id_to_coords_map) + 1))


def empty_shelf_action_ids(env) -> List[int]:
    """Return action ids for storage slots that are currently empty."""
    empty_mask = env.get_empty_shelf_information()
    return list(np.array(env.shelf_action_ids)[empty_mask > 0])


def get_requested_action_ids(env) -> List[int]:
    """Return unique action ids for all currently requested shelves."""
    inverse_map = {coords: aid for aid, coords in env.action_id_to_coords_map.items()}
    requested_ids: List[int] = []
    for shelf in env.request_queue:
        action_id = inverse_map.get((int(shelf.y), int(shelf.x)))
        if action_id is not None:
            requested_ids.append(int(action_id))
    return list(dict.fromkeys(requested_ids))


# ---------------------------------------------------------------------------
# Path and prioritization helpers
# ---------------------------------------------------------------------------
def nearest_id_by_path(env, agent, candidate_ids: List[int]) -> int:
    """Return the reachable candidate with the shortest path from the agent."""
    best_id = 0
    best_len = float("inf")
    for action_id in candidate_ids:
        coords = action_coords(env, action_id)
        if coords is None:
            continue
        path = env.find_path((agent.y, agent.x), coords, agent, care_for_agents=False)
        if path and len(path) < best_len:
            best_len = len(path)
            best_id = int(action_id)
    return best_id


def nearest_charging_station_for_agent(env, agent) -> Tuple[int, int]:
    """Return the nearest charging action id and its path length."""
    charging_ids = charging_action_ids(env)
    if not charging_ids:
        return 0, 0
    nearest_id = nearest_id_by_path(env, agent, charging_ids)
    coords = action_coords(env, nearest_id)
    if nearest_id <= 0 or coords is None:
        return 0, 0
    path = env.find_path((agent.y, agent.x), coords, agent, care_for_agents=False)
    return int(nearest_id), len(path) if path else 0


def nearest_charging_station_from_coords(env, start_coords: Tuple[int, int], agent) -> Tuple[int, int]:
    """Return the nearest charging action id and path length from arbitrary coords."""
    charging_ids = charging_action_ids(env)
    if not charging_ids:
        return 0, 0

    best_charge_id = 0
    best_steps: Optional[int] = None
    for charge_id in charging_ids:
        coords = action_coords(env, charge_id)
        if coords is None:
            continue
        path = env.find_path(start_coords, coords, agent, care_for_agents=False)
        if path is None:
            continue
        steps = len(path)
        if best_steps is None or steps < best_steps:
            best_steps = steps
            best_charge_id = int(charge_id)

    if best_steps is None:
        return 0, 0
    return best_charge_id, best_steps


def battery_need_label(battery: float) -> str:
    """Bucket a battery value into a prompt-friendly qualitative label."""
    if battery < 25.0:
        return "critical"
    if battery < 50.0:
        return "need_charging_soon"
    return "not_needed"


def candidate_ids_for_agent(env, idx: int, valid_masks: np.ndarray, max_candidate_ids: int) -> List[int]:
    """Return valid action ids sorted by prompt usefulness and proximity."""
    agent = env.agents[idx]
    valid_ids = [int(i) for i in np.where(valid_masks[idx] > 0)[0].tolist()]
    if not valid_ids:
        valid_ids = [0]

    valid_ids.sort(
        key=lambda action_id: (
            0 if action_id == 0 else 1,
            (
                len(path)
                if (
                    action_id in env.action_id_to_coords_map
                    and (
                        path := env.find_path(
                            (agent.y, agent.x),
                            env.action_id_to_coords_map[action_id],
                            agent,
                            care_for_agents=False,
                        )
                    )
                )
                else float("inf")
            ),
            action_id,
        )
    )

    if max_candidate_ids and max_candidate_ids > 0:
        return valid_ids[:max_candidate_ids]
    return valid_ids


def candidate_ids_for_prompt(env, idx: int, valid_masks: np.ndarray, max_candidate_ids: int) -> List[int]:
    """Return a prompt-sized candidate list.

    Pickers get a narrower view that still preserves charging, NOOP, and the
    shelf actions currently relevant to AGV coordination.
    """
    base_ids = candidate_ids_for_agent(env, idx, valid_masks, 0)
    if idx < env.num_agvs:
        if max_candidate_ids and max_candidate_ids > 0:
            return base_ids[:max_candidate_ids]
        return base_ids

    agv_shelf_targets = {
        int(agent.target)
        for agent in env.agents[: env.num_agvs]
        if int(agent.target) > 0 and classify_action(env, int(agent.target)) == "SHELF"
    }
    allowed = {0, *charging_action_ids(env), *agv_shelf_targets}
    filtered = [action_id for action_id in base_ids if action_id in allowed]
    if not filtered:
        filtered = base_ids

    if max_candidate_ids and max_candidate_ids > 0:
        return filtered[:max_candidate_ids]
    return filtered


# ---------------------------------------------------------------------------
# Structured state extractors
# ---------------------------------------------------------------------------
def requested_shelf_records(env) -> List[Dict[str, int]]:
    """Return requested shelves as structured records."""
    inverse_map = {coords: aid for aid, coords in env.action_id_to_coords_map.items()}
    records: List[Dict[str, int]] = []
    for shelf in env.request_queue:
        action_id = inverse_map.get((int(shelf.y), int(shelf.x)))
        records.append(
            {
                "shelf_id": int(shelf.id),
                "x": int(shelf.x),
                "y": int(shelf.y),
                "action_id": int(action_id) if action_id is not None else -1,
            }
        )
    return records


def agent_target_text(env, agent) -> str:
    """Describe the agent's current target in a compact prompt form."""
    target_id = int(agent.target)
    coords = action_coords(env, target_id) if target_id > 0 else None
    if coords is None:
        return "none"
    return f"{target_id}:{classify_action(env, target_id)}@({int(coords[1])},{int(coords[0])})"


def self_state_record(env, agent) -> Dict[str, Any]:
    """Return a structured self-state snapshot for one agent."""
    return {
        "x": int(agent.x),
        "y": int(agent.y),
        "target": agent_target_text(env, agent),
        "busy": bool(agent.busy),
        "battery": float(agent.battery),
        "battery_need": battery_need_label(float(agent.battery)),
        "carrying": int(agent.carrying_shelf.id) if agent.carrying_shelf else None,
    }


def all_agent_records(env) -> List[Dict[str, Any]]:
    """Return a compact structured snapshot for all agents."""
    records: List[Dict[str, Any]] = []
    for idx, agent in enumerate(env.agents):
        records.append(
            {
                "agent_id": idx,
                "agent_type": agent.type.name,
                "x": int(agent.x),
                "y": int(agent.y),
                "target": agent_target_text(env, agent),
                "busy": bool(agent.busy),
                "battery": float(agent.battery),
                "carrying": int(agent.carrying_shelf.id) if agent.carrying_shelf else None,
            }
        )
    return records


# ---------------------------------------------------------------------------
# Higher-level natural-language renderers
# ---------------------------------------------------------------------------
def describe_action_id_for_agent(env, agent, action_id: int) -> str:
    """Translate a candidate action id into a readable prompt line."""
    if int(action_id) == 0:
        return "0:NOOP (distance_steps=0)"

    action_type = classify_action(env, int(action_id))
    coords = action_coords(env, int(action_id))
    if coords is None:
        return f"{int(action_id)}:{action_type}"

    path = env.find_path((agent.y, agent.x), coords, agent, care_for_agents=False)
    distance = len(path) if path else 0
    return f"{int(action_id)}:{action_type}@({int(coords[1])},{int(coords[0])}) (distance_steps={distance})"


def get_requested_shelves(env) -> List[str]:
    """Render requested shelves as natural-language prompt lines."""
    lines: List[str] = []
    for record in requested_shelf_records(env):
        if record["action_id"] < 0:
            lines.append(
                f"Requested shelf {record['shelf_id']} is at position ({record['x']},{record['y']}), "
                "but no action id was found."
            )
            continue
        lines.append(
            f"Requested shelf {record['shelf_id']} is at position ({record['x']},{record['y']}). "
            f"To go there, choose action id {record['action_id']}."
        )
    return lines


def get_requested_shelves_for_agent(env, agent) -> List[str]:
    """Render requested shelves ranked by path distance for one agent."""
    rows: List[Tuple[int, int, int, int, int]] = []
    for record in requested_shelf_records(env):
        if record["action_id"] < 0:
            continue
        coords = action_coords(env, record["action_id"])
        if coords is None:
            continue
        path = env.find_path((agent.y, agent.x), coords, agent, care_for_agents=False)
        if path is None:
            continue
        rows.append((len(path), record["shelf_id"], record["action_id"], record["x"], record["y"]))

    rows.sort(key=lambda row: (row[0], row[2]))
    return [
        f"shelf_id={shelf_id} action_id={action_id} pos=({x},{y}) distance_steps={steps}"
        for steps, shelf_id, action_id, x, y in rows
    ]


def get_picker_agv_task_view(env) -> List[str]:
    """Render AGV support demand in a picker-friendly summary."""
    requested_shelf_ids = {int(shelf.id) for shelf in env.request_queue}
    picker_positions = {(int(agent.x), int(agent.y)) for agent in env.agents if agent.type.name == "PICKER"}
    lines: List[str] = []

    for idx, agv in enumerate(env.agents[: env.num_agvs]):
        target_id = int(agv.target)
        coords = action_coords(env, target_id) if target_id > 0 else None
        at_target = bool(coords is not None and (int(agv.x), int(agv.y)) == (int(coords[1]), int(coords[0])))
        carrying = agv.carrying_shelf is not None
        carrying_requested = bool(carrying and int(agv.carrying_shelf.id) in requested_shelf_ids)
        target_type = classify_action(env, target_id) if target_id > 0 else "NOOP"
        picker_here = bool((int(agv.x), int(agv.y)) in picker_positions)

        support_need = "none"
        if target_type == "SHELF" and at_target and not picker_here:
            support_need = "needs_picker_now_at_target"
        elif target_type == "SHELF" and not at_target and not carrying and target_id in get_requested_action_ids(env):
            support_need = "may_need_picker_soon_for_load"
        elif target_type == "SHELF" and not at_target and carrying and not carrying_requested:
            support_need = "may_need_picker_soon_for_unload"

        lines.append(
            f"agv agent_{idx}: target={target_id}:{target_type} "
            f"busy={bool(agv.busy)} carrying={int(agv.carrying_shelf.id) if carrying else None} "
            f"battery={float(agv.battery):.1f} support_need={support_need}"
        )
    return lines


def render_self_state(env, agent) -> str:
    """Render one self-state record as a natural-language line."""
    state = self_state_record(env, agent)
    return (
        f"pos=({state['x']},{state['y']}), "
        f"target={state['target']}, "
        f"busy={state['busy']}, "
        f"battery={state['battery']:.1f} ({state['battery_need']}), "
        f"carrying={state['carrying']}"
    )


def render_all_agents(env) -> List[str]:
    """Render all agents as natural-language summary lines."""
    rows: List[str] = []
    for record in all_agent_records(env):
        rows.append(
            f"agent_{record['agent_id']} type={record['agent_type']} pos=({record['x']},{record['y']}) "
            f"target={record['target']} busy={record['busy']} battery={record['battery']:.1f} "
            f"carrying={record['carrying']}"
        )
    return rows


def charging_station_occupancy(env) -> List[str]:
    """Render charging-station occupancy as compact natural-language lines."""
    lines: List[str] = []
    for action_id in charging_action_ids(env):
        coords = action_coords(env, action_id)
        if coords is None:
            continue
        x, y = int(coords[1]), int(coords[0])
        occupants: List[str] = []
        for idx, agent in enumerate(env.agents):
            if (int(agent.x), int(agent.y)) == (x, y):
                occupants.append(
                    f"agent_{idx}({agent.type.name},battery={float(agent.battery):.1f},need={battery_need_label(float(agent.battery))})"
                )
        lines.append(f"charging_action={action_id} at ({x},{y}) occupants=[{', '.join(occupants) if occupants else 'none'}]")
    return lines


def waiting_agv_support_lines(env) -> List[str]:
    """Render AGVs that currently need picker support at shelf interactions."""
    lines: List[str] = []
    picker_positions = {(int(agent.x), int(agent.y)) for agent in env.agents if agent.type.name == "PICKER"}

    for idx, agv in enumerate(env.agents[: env.num_agvs]):
        target_id = int(agv.target)
        if target_id <= 0 or classify_action(env, target_id) != "SHELF":
            continue

        coords = action_coords(env, target_id)
        if coords is None:
            continue

        at_target = (int(agv.x), int(agv.y)) == (int(coords[1]), int(coords[0]))
        picker_here = (int(agv.x), int(agv.y)) in picker_positions
        carrying = agv.carrying_shelf is not None

        if not bool(agv.busy):
            continue
        if at_target and not picker_here:
            need = "waiting_for_picker_support_now"
        elif not at_target and not carrying:
            need = "moving_to_shelf_will_need_picker_for_load"
        elif not at_target and carrying:
            need = "moving_to_shelf_will_need_picker_for_unload"
        else:
            continue

        lines.append(
            f"agent_{idx} target={target_id}:{classify_action(env, target_id)}@({int(coords[1])},{int(coords[0])}) "
            f"pos=({int(agv.x)},{int(agv.y)}) carrying={int(agv.carrying_shelf.id) if carrying else None} "
            f"support_need={need}"
        )
    return lines
