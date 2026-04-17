import json
import re
from unittest.mock import patch

import numpy as np

from tarware.definitions import Action
from tarware.warehouse import RewardType, Warehouse

from scripts.run_obj2_shared_context_llm import (
    ScenarioSpec,
    agv_requires_protected_unload_wait,
    agent_is_interruptible_when_busy,
    agent_is_interruptible_on_charger,
    build_agent_prompt,
    effective_agv_support_rows,
    effective_picker_support_rows,
    next_required_flow_state,
    obj2_candidate_action_ids,
    order_agents_for_planning,
    parse_steps_from_text,
    picker_task_ready_charge_policy,
    picker_support_candidate_action_lines,
    parser,
    plan_step_sequential,
    persisted_action_completed_by_env,
    run_single_episode,
    safe_valid_action_masks,
    support_flags_for_agv,
)


def make_env() -> Warehouse:
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
        allow_busy_replan=True,
    )
    env.reset(seed=0)
    return env


def action_id_for_coords(env: Warehouse, x: int, y: int) -> int:
    for action_id, coords in env.action_id_to_coords_map.items():
        if (int(coords[1]), int(coords[0])) == (x, y):
            return int(action_id)
    raise AssertionError(f"No action found at {(x, y)}")


def extract_prompt_json(prompt: str) -> dict:
    match = re.search(
        r"Shared planning context in JSON:\n(.*?)\n\nThe planning context ends above\.",
        prompt,
        flags=re.DOTALL,
    )
    if match is None:
        raise AssertionError("No embedded planning JSON found in prompt")
    payload = json.loads(match.group(1))
    if "planning_context" not in payload and "current_control_state" in payload:
        candidate_actions = payload.get("candidate_actions", [])
        role_context = payload.get("role_context", {})
        current_control_state = payload.get("current_control_state", {})
        if payload.get("current_agent", {}).get("agent_type") == "AGV":
            role_specific_context = {
                "agv_state": {
                    "phase": role_context.get("phase"),
                    "carrying_status": role_context.get("carrying_status"),
                    "carrying_delivery_status": role_context.get("carrying_delivery_status"),
                    "battery_need": role_context.get("battery_need"),
                    "current_target_type": role_context.get("current_target_type"),
                    "current_target_distance_steps": role_context.get("current_target_distance_steps"),
                    "goal_allowed": role_context.get("goal_allowed"),
                    "empty_shelf_return_required": role_context.get("empty_shelf_return_required"),
                    "preferred_action_family": role_context.get("preferred_action_family"),
                    "forbidden_action_family": role_context.get("forbidden_action_family"),
                    "next_required_flow_state": current_control_state.get("next_required_flow_state"),
                },
                "support_flags": {
                    "waiting_for_picker_support": current_control_state.get("waiting_for_picker_support_now"),
                    "picker_at_same_cell": current_control_state.get("picker_at_same_cell"),
                    "picker_support_inbound_now": current_control_state.get("picker_support_inbound_now"),
                    "picker_support_distance_steps": current_control_state.get("picker_support_distance_steps"),
                    "needs_picker_for_load": current_control_state.get("needs_picker_for_load"),
                    "needs_picker_for_unload": current_control_state.get("needs_picker_for_unload"),
                },
                "goal_delivery_context": role_context.get("goal_delivery_context", {}),
                "empty_shelf_return_context": role_context.get("empty_shelf_return_context", {}),
                "charging_context": role_context.get("charging_context", {}),
            }
            if "post_delivery_helper" in role_context:
                role_specific_context["post_delivery_helper"] = role_context["post_delivery_helper"]
            if "post_load_helper" in role_context:
                role_specific_context["post_load_helper"] = role_context["post_load_helper"]
            if "empty_shelf_return_task" in role_context:
                role_specific_context["empty_shelf_return_task"] = role_context["empty_shelf_return_task"]
        else:
            role_specific_context = {
                "picker_support_context": {
                    "phase": role_context.get("phase"),
                    "battery_need": role_context.get("battery_need"),
                    "support_needed_now": role_context.get("support_needed_now"),
                    "charging_allowed_now": role_context.get("charging_allowed_now"),
                    "support_target_action_id": role_context.get("support_target_action_id"),
                    "support_target_position": role_context.get("support_target_position"),
                    "support_type": role_context.get("support_type"),
                    "at_support_target": role_context.get("at_support_target"),
                    "agv_waiting_for_picker_now": role_context.get("agv_waiting_for_picker_now"),
                    "next_required_flow_state": current_control_state.get("next_required_flow_state"),
                    "agv_support_timing_view": role_context.get("agv_support_timing_view", []),
                },
                "picker_support_candidate_actions": role_context.get("picker_support_candidate_actions", []),
            }
            if "post_delivery_helper" in role_context:
                role_specific_context["post_delivery_helper"] = role_context["post_delivery_helper"]
            if "post_load_helper" in role_context:
                role_specific_context["post_load_helper"] = role_context["post_load_helper"]
        payload["planning_context"] = {
            "shared_coordination_context": payload.get("shared_coordination_context", {}),
            "current_agent": {
                **payload.get("current_agent", {}),
                "next_required_flow_state": payload.get("current_control_state", {}).get("next_required_flow_state"),
                "control_state": payload.get("current_control_state", {}),
                "last_decided_action": payload.get("last_decided_action_state"),
                "movement_warning": payload.get("movement_warning"),
                "blocking_warning": payload.get("blocking_warning"),
                "execution_warning": payload.get("execution_warning"),
                "candidate_actions": [
                    {
                        "action_id": row.get("action"),
                        "action_type": row.get("type"),
                        "target_position": row.get("position"),
                        "distance_steps": row.get("distance_steps"),
                        "suggested_steps": row.get("suggested_steps"),
                        "tag": row.get("tag"),
                        "instruction": row.get("instruction"),
                        "recommended_now": row.get("recommended_now"),
                    }
                    for row in candidate_actions
                ],
                "primary_candidate_actions": [
                    {
                        "action_id": row.get("action"),
                        "action_type": row.get("type"),
                        "target_position": row.get("position"),
                        "distance_steps": row.get("distance_steps"),
                        "suggested_steps": row.get("suggested_steps"),
                        "tag": row.get("tag"),
                        "instruction": row.get("instruction"),
                        "recommended_now": row.get("recommended_now"),
                    }
                    for row in candidate_actions
                    if row.get("type") != "CHARGING"
                ],
                "charging_candidate_header": (
                    "IF CHARGING is needed, select following actions:"
                    if any(row.get("type") == "CHARGING" for row in candidate_actions)
                    else None
                ),
                "charging_candidate_actions": [
                    {
                        "action_id": row.get("action"),
                        "action_type": row.get("type"),
                        "target_position": row.get("position"),
                        "distance_steps": row.get("distance_steps"),
                        "suggested_steps": row.get("suggested_steps"),
                        "tag": row.get("tag"),
                        "instruction": row.get("instruction"),
                        "recommended_now": row.get("recommended_now"),
                    }
                    for row in candidate_actions
                    if row.get("type") == "CHARGING"
                ],
            },
            "role_specific_context": role_specific_context,
        }
    return payload


def test_agv_language_prompt_includes_rich_flow_and_phase() -> None:
    env = make_env()
    try:
        agv = env.agents[0]
        requested_shelf = env.request_queue[0]
        agv.carrying_shelf = requested_shelf
        agv.busy = True
        agv.target = 1
        valid_masks = safe_valid_action_masks(env)

        prompt = build_agent_prompt(env, 0, valid_masks, "language", 10)
        assert "GOAL marks delivery only; AGV still carries the same shelf after delivery." in prompt
        assert "After GOAL delivery, AGV should return the carried shelf to an EMPTY shelf location." in prompt
        assert "phase=carrying_requested_shelf_to_goal" in prompt
        assert "Current control state:" in prompt
        assert "next_required_flow_state: move_requested_shelf_to_goal" in prompt
        assert "Carrying status: carrying_requested_shelf" in prompt
        assert "Post-load helper:" in prompt
        assert "Load already happened at the shelf." in prompt
        assert "Do not choose the same shelf again." in prompt
        assert "Your current task is to move this shelf to GOAL." in prompt
        assert "AGV flow rules:" in prompt
        assert "If carrying a requested shelf: choose GOAL, not shelf." in prompt
        assert "If load already happened and you are carrying the requested shelf, do not choose the load shelf again; choose GOAL." in prompt
        assert "Decision contract:" in prompt
        assert "Do not write recommendations." in prompt
        assert "Return exactly 3 lines and nothing else." in prompt
    finally:
        env.close()


def test_picker_language_prompt_includes_support_context_without_redundant_header() -> None:
    env = make_env()
    try:
        agv = env.agents[0]
        requested_shelf = env.request_queue[0]
        target_action = action_id_for_coords(env, int(requested_shelf.x), int(requested_shelf.y))
        agv.target = target_action
        agv.busy = True
        agv.carrying_shelf = None
        valid_masks = safe_valid_action_masks(env)

        prompt = build_agent_prompt(env, 1, valid_masks, "language", 3)
        assert "Picker support candidate shelf actions:" in prompt
        assert f"{target_action}:SHELF" in prompt
        assert "support_need=" in prompt
        assert "Picker support context:" in prompt
        assert "Current-state hard constraints:" in prompt
        assert "Picker support should be at the same shelf action id as the AGV support target, not at GOAL." in prompt
        assert "Do not enter a shelf cell unless the AGV is already there for support; otherwise stay near the shelf and wait." in prompt
        assert "AGVs requesting picker support:" not in prompt
        assert "Current control state:" in prompt
        assert "next_required_flow_state: move_to_support_load" in prompt
    finally:
        env.close()


def test_agv_json_prompt_contains_phase_and_agv_role_specific_context() -> None:
    env = make_env()
    try:
        agv = env.agents[0]
        requested_shelf = env.request_queue[0]
        agv.carrying_shelf = requested_shelf
        agv.busy = True
        agv.target = 1
        valid_masks = safe_valid_action_masks(env)

        prompt = build_agent_prompt(env, 0, valid_masks, "json", 10)
        full_payload = extract_prompt_json(prompt)
        payload = full_payload["planning_context"]
        assert payload["current_agent"]["phase"] == "carrying_requested_shelf_to_goal"
        assert "target_type" in payload["current_agent"]
        assert payload["current_agent"]["next_required_flow_state"] == "move_requested_shelf_to_goal"
        assert full_payload["current_control_state"]["next_required_flow_state"] == "move_requested_shelf_to_goal"
        assert full_payload["current_control_state"]["preferred_action_family"] == "goal"
        assert isinstance(full_payload["candidate_actions"], list)
        assert isinstance(full_payload["decision_rules"], list)
        assert "control_state" in payload["current_agent"]
        assert payload["current_agent"]["control_state"]["load_completed_and_goal_required_now"] is True
        assert payload["current_agent"]["control_state"]["recommended_goal_action_id"] is not None
        assert payload["role_specific_context"]["post_load_helper"]["current_post_load_goal_action_id"] is not None
        assert "agv_state" in payload["role_specific_context"]
        assert payload["role_specific_context"]["post_load_helper"]["post_load_active"] is True
        assert "goal_delivery_context" in payload["role_specific_context"]
        assert "empty_shelf_return_context" in payload["role_specific_context"]
        assert "movement_steps_hint" in payload["steps_guidance"]
        assert "Action must be an integer action id only" in prompt
    finally:
        env.close()


def test_picker_json_prompt_contains_phase_and_picker_support_context() -> None:
    env = make_env()
    try:
        agv = env.agents[0]
        requested_shelf = env.request_queue[0]
        target_action = action_id_for_coords(env, int(requested_shelf.x), int(requested_shelf.y))
        agv.target = target_action
        agv.busy = True
        agv.carrying_shelf = None
        valid_masks = safe_valid_action_masks(env)

        prompt = build_agent_prompt(env, 1, valid_masks, "json", 3)
        full_payload = extract_prompt_json(prompt)
        payload = full_payload["planning_context"]
        assert "phase" in payload["current_agent"]
        assert "target_type" in payload["current_agent"]
        assert payload["current_agent"]["next_required_flow_state"] == "move_to_support_load"
        assert full_payload["current_control_state"]["next_required_flow_state"] == "move_to_support_load"
        assert full_payload["current_control_state"]["support_needed_now"] is True
        assert isinstance(full_payload["candidate_actions"], list)
        assert "picker_support_context" in payload["role_specific_context"]
        assert "picker_support_candidate_actions" in payload["role_specific_context"]
        assert "support_needed_now" in payload["role_specific_context"]["picker_support_context"]
        assert "agv_support_timing_view" in payload["role_specific_context"]["picker_support_context"]
        assert "Action must be an integer action id only" in prompt
    finally:
        env.close()


def test_agv_candidate_actions_are_pruned_to_relevant_choices() -> None:
    env = make_env()
    try:
        agv = env.agents[0]
        requested_shelf = env.request_queue[0]
        target_action = action_id_for_coords(env, int(requested_shelf.x), int(requested_shelf.y))
        valid_masks = safe_valid_action_masks(env)

        prompt = build_agent_prompt(env, 0, valid_masks, "language", 10)
        assert f"{target_action}:SHELF" in prompt
        assert "CHARGING@" in prompt
        assert "GOAL@" not in prompt
        assert "0:NOOP" not in prompt
        assert "If battery_need=not_needed and next_required_flow_state is not charge_now, charging is forbidden." in prompt
        assert "carrying=None => GOAL action ids are forbidden." in prompt
        assert "carrying=None => choose a requested shelf action." in prompt
        assert "prefer the closest candidate by distance_steps." in prompt
        assert "suggested_steps=" in prompt
        assert "tag=requested_shelf" in prompt
        assert "IF CHARGING is needed, select following actions:" in prompt
        assert "instruction=forbidden_now_do_not_choose_charging" in prompt

        agv.carrying_shelf = requested_shelf
        prompt = build_agent_prompt(env, 0, valid_masks, "language", 10)
        assert "GOAL@" in prompt
        assert "0:NOOP" not in prompt
        assert "carrying=requested shelf => GOAL action ids are allowed and preferred." in prompt
        assert "instruction=choose_this_to_deliver_to_goal" in prompt
        assert "recommended_now=yes" in prompt
    finally:
        env.close()


def test_prompt_includes_last_decided_action_state() -> None:
    env = make_env()
    try:
        valid_masks = safe_valid_action_masks(env)
        requested_shelf = env.request_queue[0]
        target_action = action_id_for_coords(env, int(requested_shelf.x), int(requested_shelf.y))

        prompt = build_agent_prompt(env, 0, valid_masks, "language", 10, persisted_action=target_action, remaining_hold_steps=0)
        assert "Last decided action state:" in prompt
        assert f"Last decided action: {target_action}:SHELF" in prompt
        assert "Remaining hold steps: 0" in prompt
        assert "Requery required now: True" in prompt

        json_prompt = build_agent_prompt(env, 0, valid_masks, "json", 10, persisted_action=target_action, remaining_hold_steps=2)
        payload = extract_prompt_json(json_prompt)["planning_context"]
        last_decided = payload["current_agent"]["last_decided_action"]
        assert last_decided["last_decided_action_id"] == target_action
        assert "SHELF@" in last_decided["last_decided_action_description"]
        assert last_decided["remaining_hold_steps"] == 2
        assert last_decided["requery_required_now"] is False
    finally:
        env.close()


def test_movement_warning_appears_when_agent_not_moved_for_positive_steps() -> None:
    env = make_env()
    try:
        valid_masks = safe_valid_action_masks(env)
        requested_shelf = env.request_queue[0]
        target_action = action_id_for_coords(env, int(requested_shelf.x), int(requested_shelf.y))
        env.stuck_counters[0].count = 4

        prompt = build_agent_prompt(env, 0, valid_masks, "language", 10, persisted_action=target_action, remaining_hold_steps=0)
        assert "Movement warning:" in prompt
        assert "You have not moved for 4 steps." in prompt
        assert "If you have not moved for long, consider a different target." in prompt
        assert f"Your previous target was: {target_action}:SHELF" in prompt
    finally:
        env.close()


def test_movement_warning_is_omitted_when_not_moved_steps_is_zero() -> None:
    env = make_env()
    try:
        valid_masks = safe_valid_action_masks(env)
        prompt = build_agent_prompt(env, 1, valid_masks, "language", 3)
        assert "Movement warning:" not in prompt
        assert "You have not moved for" not in prompt
    finally:
        env.close()


def test_json_prompt_includes_movement_warning_payload_from_last_decided_action() -> None:
    env = make_env()
    try:
        valid_masks = safe_valid_action_masks(env)
        requested_shelf = env.request_queue[0]
        target_action = action_id_for_coords(env, int(requested_shelf.x), int(requested_shelf.y))
        env.stuck_counters[1].count = 3

        payload = extract_prompt_json(
            build_agent_prompt(env, 1, valid_masks, "json", 3, persisted_action=target_action, remaining_hold_steps=0)
        )["planning_context"]
        warning = payload["current_agent"]["movement_warning"]
        assert warning["not_moved_steps"] == 3
        assert warning["consider_different_target_if_long"] is True
        assert warning["previous_target_action_id"] == target_action
        assert "SHELF@" in warning["previous_target_description"]
    finally:
        env.close()


def test_env_info_includes_blocking_agent_id_and_reason_for_adjacent_block_case() -> None:
    env = make_env()
    try:
        requested_shelf = env.request_queue[0]
        target_action = action_id_for_coords(env, int(requested_shelf.x), int(requested_shelf.y))
        agv = env.agents[0]
        picker = env.agents[1]
        agv.x = 5
        agv.y = 4
        agv.busy = True
        agv.target = target_action
        agv.req_action = Action.NOOP
        picker.x = 6
        picker.y = 4
        env._agents_in_conflict = {agv.id}

        info = env._build_info(0, 0, 0, 0, 0, [target_action, 0])
        assert info["blocking_agent_id_by_agent"][0] == 1
        assert info["blocking_reason_by_agent"][0] == "picker_blocking_agv_path"
    finally:
        env.close()


def test_movement_warning_uses_last_decided_action_not_current_env_target() -> None:
    env = make_env()
    try:
        valid_masks = safe_valid_action_masks(env)
        requested_shelf = env.request_queue[0]
        previous_target = action_id_for_coords(env, int(requested_shelf.x), int(requested_shelf.y))
        current_target = next(action_id for action_id in env.shelf_action_ids if int(action_id) != previous_target)
        env.agents[0].target = current_target
        env.stuck_counters[0].count = 2

        prompt = build_agent_prompt(env, 0, valid_masks, "language", 10, persisted_action=previous_target, remaining_hold_steps=0)
        assert f"Your previous target was: {previous_target}:SHELF" in prompt
        assert f"Your previous target was: {current_target}:SHELF" not in prompt
    finally:
        env.close()


def test_agv_prompt_includes_blocked_by_agent_warning() -> None:
    env = make_env()
    try:
        valid_masks = safe_valid_action_masks(env)
        requested_shelf = env.request_queue[0]
        target_action = action_id_for_coords(env, int(requested_shelf.x), int(requested_shelf.y))
        agv = env.agents[0]
        picker = env.agents[1]
        agv.x = 5
        agv.y = 4
        agv.busy = True
        agv.target = target_action
        agv.req_action = Action.NOOP
        picker.x = 6
        picker.y = 4
        env._agents_in_conflict = {agv.id}

        prompt = build_agent_prompt(env, 0, valid_masks, "language", 10)
        assert "Blocking warning:" in prompt
        assert "You are currently blocked by agent_1." in prompt
        assert "Reason: picker_blocking_agv_path." in prompt
    finally:
        env.close()


def test_picker_prompt_includes_may_be_blocking_warning() -> None:
    env = make_env()
    try:
        valid_masks = safe_valid_action_masks(env)
        requested_shelf = env.request_queue[0]
        target_action = action_id_for_coords(env, int(requested_shelf.x), int(requested_shelf.y))
        agv = env.agents[0]
        picker = env.agents[1]
        agv.x = 5
        agv.y = 4
        agv.busy = True
        agv.target = target_action
        agv.req_action = Action.NOOP
        picker.x = 6
        picker.y = 4
        env._agents_in_conflict = {agv.id}

        prompt = build_agent_prompt(env, 1, valid_masks, "language", 3)
        assert "Blocking warning:" in prompt
        assert "You may be blocking agent_0." in prompt
        assert "Reason: picker_blocking_agv_path." in prompt
    finally:
        env.close()


def test_json_prompt_includes_blocking_warning_payload() -> None:
    env = make_env()
    try:
        valid_masks = safe_valid_action_masks(env)
        requested_shelf = env.request_queue[0]
        target_action = action_id_for_coords(env, int(requested_shelf.x), int(requested_shelf.y))
        agv = env.agents[0]
        picker = env.agents[1]
        agv.x = 5
        agv.y = 4
        agv.busy = True
        agv.target = target_action
        agv.req_action = Action.NOOP
        picker.x = 6
        picker.y = 4
        env._agents_in_conflict = {agv.id}

        payload = extract_prompt_json(build_agent_prompt(env, 1, valid_masks, "json", 3))["planning_context"]
        warning = payload["current_agent"]["blocking_warning"]
        assert warning["blocking_agent_id"] == 0
        assert warning["blocked_by_agent_id"] is None
        assert warning["reason"] == "picker_blocking_agv_path"
    finally:
        env.close()


def test_picker_prompt_shows_blocking_warning_when_blocking_agent_0() -> None:
    env = make_env()
    try:
        valid_masks = safe_valid_action_masks(env)
        requested_shelf = env.request_queue[0]
        target_action = action_id_for_coords(env, int(requested_shelf.x), int(requested_shelf.y))
        agv = env.agents[0]
        picker = env.agents[1]
        agv.x = 5
        agv.y = 4
        agv.busy = True
        agv.target = target_action
        agv.req_action = Action.NOOP
        picker.x = 6
        picker.y = 4
        env._agents_in_conflict = {agv.id}

        prompt = build_agent_prompt(env, 1, valid_masks, "language", 3)
        assert "Blocking warning:" in prompt
        assert "You may be blocking agent_0." in prompt

        payload = extract_prompt_json(build_agent_prompt(env, 1, valid_masks, "json", 3))["planning_context"]
        assert payload["current_agent"]["blocking_warning"]["blocking_agent_id"] == 0
    finally:
        env.close()


def test_prompt_omits_blocking_warning_when_no_blocker_is_reported() -> None:
    env = make_env()
    try:
        valid_masks = safe_valid_action_masks(env)
        prompt = build_agent_prompt(env, 0, valid_masks, "language", 10)
        assert "Blocking warning:" not in prompt
    finally:
        env.close()


def test_busy_override_warning_appears_in_language_prompt_when_force_replanning_busy_agent() -> None:
    env = make_env()
    try:
        valid_masks = safe_valid_action_masks(env)
        requested_shelf = env.request_queue[0]
        target_action = action_id_for_coords(env, int(requested_shelf.x), int(requested_shelf.y))
        prompt = build_agent_prompt(
            env,
            0,
            valid_masks,
            "language",
            10,
            persisted_action=target_action,
            remaining_hold_steps=0,
            busy_steps_by_agent=[6, 0],
            busy_override_active=True,
        )
        assert "Busy override warning: You have remained busy for 6 steps, so you are being replanned now." in prompt
        assert f"Your previous target was: {target_action}:SHELF" in prompt
    finally:
        env.close()


def test_busy_override_warning_payload_appears_in_json_prompt() -> None:
    env = make_env()
    try:
        valid_masks = safe_valid_action_masks(env)
        requested_shelf = env.request_queue[0]
        target_action = action_id_for_coords(env, int(requested_shelf.x), int(requested_shelf.y))
        payload = extract_prompt_json(
            build_agent_prompt(
                env,
                1,
                valid_masks,
                "json",
                3,
                persisted_action=target_action,
                remaining_hold_steps=0,
                busy_steps_by_agent=[0, 4],
                busy_override_active=True,
            )
        )["planning_context"]
        warning = payload["current_agent"]["movement_warning"]
        assert warning["busy_override_active"] is True
        assert warning["busy_steps_count"] == 4
        assert warning["previous_target_action_id"] == target_action
    finally:
        env.close()


def test_picker_prompt_forbids_algorithmic_output() -> None:
    env = make_env()
    try:
        agv = env.agents[0]
        requested_shelf = env.request_queue[0]
        target_action = action_id_for_coords(env, int(requested_shelf.x), int(requested_shelf.y))
        agv.target = target_action
        agv.busy = True
        valid_masks = safe_valid_action_masks(env)

        prompt = build_agent_prompt(env, 1, valid_masks, "language", 3)
        assert "Single-step decision:" in prompt
        assert "Do not propose a strategy for future behavior." in prompt
        assert "Decision contract:" in prompt
        assert "Do not write recommendations." in prompt
        assert "Do not rewrite flow rules." in prompt
        assert "Do not give example output." in prompt
        assert "Do not output headings or bullet lists." in prompt
        assert "Action must be the integer action id only." in prompt
        assert "Steps must be an integer only." in prompt
        assert "Valid example: Action: 56" in prompt
        assert "Invalid example: Action: CHARGING@(1,0)" in prompt
        assert "Reason: <one short sentence>" in prompt
    finally:
        env.close()


def test_picker_prompt_support_now_overrides_charging_when_battery_not_critical() -> None:
    env = make_env()
    try:
        picker = env.agents[1]
        picker.x = 2
        picker.y = 0
        picker.battery = 52.0

        agv = env.agents[0]
        requested_shelf = env.request_queue[0]
        target_action = action_id_for_coords(env, int(requested_shelf.x), int(requested_shelf.y))
        agv.target = target_action
        agv.busy = True
        agv.carrying_shelf = None

        valid_masks = safe_valid_action_masks(env)
        prompt = build_agent_prompt(env, 1, valid_masks, "language", 3)
        assert "Support needed now: True" in prompt
        assert "Battery need: not_needed" in prompt
        assert "AGV support is needed now => choose the matching support shelf action, not charging." in prompt
        assert "Battery is not critical => charging is forbidden right now." in prompt
        assert "If AGV support is needed now and battery is not critical, do not choose charging." in prompt
        assert "state_source=live_env" in prompt

        json_prompt = build_agent_prompt(env, 1, valid_masks, "json", 3)
        payload = extract_prompt_json(json_prompt)["planning_context"]
        picker_context = payload["role_specific_context"]["picker_support_context"]
        assert picker_context["support_needed_now"] is True
        assert picker_context["battery_need"] == "not_needed"
        assert picker_context["charging_allowed_now"] is False
    finally:
        env.close()


def test_picker_support_uses_persisted_action_when_live_target_is_cleared() -> None:
    env = make_env()
    try:
        requested_shelf = env.request_queue[0]
        target_action = action_id_for_coords(env, int(requested_shelf.x), int(requested_shelf.y))

        agv = env.agents[0]
        agv.x = int(requested_shelf.x)
        agv.y = int(requested_shelf.y)
        agv.target = 0
        agv.busy = False
        agv.carrying_shelf = None

        valid_masks = safe_valid_action_masks(env)
        prompt = build_agent_prompt(
            env,
            1,
            valid_masks,
            "language",
            3,
            persisted_actions_all=[target_action, 0],
            hold_steps_remaining_all=[0, 0],
        )
        assert f"{target_action}:SHELF" in prompt
        assert "waiting_for_picker_support_now" in prompt
        assert "state_source=persisted_action" in prompt
    finally:
        env.close()


def test_picker_support_includes_unload_support_when_agv_is_carrying() -> None:
    env = make_env()
    try:
        agv = env.agents[0]
        delivered_shelf = env.shelfs[0]
        agv.carrying_shelf = delivered_shelf
        unload_action = action_id_for_coords(env, 2, 2)
        agv.target = unload_action
        agv.busy = True

        valid_masks = safe_valid_action_masks(env)
        prompt = build_agent_prompt(env, 1, valid_masks, "language", 3)
        assert f"{unload_action}:SHELF" in prompt
        assert "moving_to_shelf_will_need_picker_for_unload" in prompt
    finally:
        env.close()


def test_picker_support_not_raised_when_picker_already_on_same_shelf() -> None:
    env = make_env()
    try:
        agv = env.agents[0]
        picker = env.agents[1]
        requested_shelf = env.request_queue[0]
        target_action = action_id_for_coords(env, int(requested_shelf.x), int(requested_shelf.y))
        agv.target = target_action
        agv.busy = True
        agv.carrying_shelf = None
        agv.x = int(requested_shelf.x)
        agv.y = int(requested_shelf.y)
        picker.x = int(requested_shelf.x)
        picker.y = int(requested_shelf.y)

        valid_masks = safe_valid_action_masks(env)
        prompt = build_agent_prompt(env, 1, valid_masks, "language", 3)
        assert "Picker support candidate shelf actions:" in prompt
        assert "requested_by=agent_0" not in prompt
    finally:
        env.close()


def test_agv_json_prompt_exposes_goal_allowed_flag_for_non_carrying_state() -> None:
    env = make_env()
    try:
        valid_masks = safe_valid_action_masks(env)
        prompt = build_agent_prompt(env, 0, valid_masks, "json", 10)
        assert "current_control_state.next_required_flow_state" in prompt
        assert 'Use only the keys reason, action, and steps.' in prompt

        full_payload = extract_prompt_json(prompt)
        payload = full_payload["planning_context"]
        agv_state = payload["role_specific_context"]["agv_state"]
        assert agv_state["carrying_status"] == "not_carrying"
        assert agv_state["goal_allowed"] is False
        assert agv_state["preferred_action_family"] == "requested_shelf"
        assert agv_state["forbidden_action_family"] == "goal"
        assert payload["current_agent"]["next_required_flow_state"] == "move_to_requested_shelf"
        assert full_payload["current_control_state"]["next_required_flow_state"] == "move_to_requested_shelf"
        assert full_payload["current_control_state"]["preferred_action_family"] == "requested_shelf"
    finally:
        env.close()


def test_agv_waiting_for_picker_prompt_says_hold_position() -> None:
    env = make_env()
    try:
        agv = env.agents[0]
        picker = env.agents[1]
        requested_shelf = env.request_queue[0]
        target_action = action_id_for_coords(env, int(requested_shelf.x), int(requested_shelf.y))
        agv.target = target_action
        agv.busy = True
        agv.carrying_shelf = None
        agv.x = int(requested_shelf.x)
        agv.y = int(requested_shelf.y)
        picker.target = target_action
        picker.busy = True
        picker.x = max(0, int(requested_shelf.x) - 1)
        picker.y = int(requested_shelf.y)

        valid_masks = safe_valid_action_masks(env)
        prompt = build_agent_prompt(env, 0, valid_masks, "language", 10)
        assert "phase=waiting_for_picker_at_requested_shelf" in prompt
        assert "The AGV has already reached the requested shelf." in prompt
        assert "Picker support has been raised for this shelf and Picker should be moving here." in prompt
        assert "Hold position and wait unless battery is critical or another very important action is required." in prompt
        assert "picker_support_inbound_now: yes" in prompt
        assert "Picker is on the way to this same shelf now." in prompt
        assert "Do not leave this shelf now." in prompt
        assert "waiting_for_picker_support_now: yes" in prompt
        assert "next_required_flow_state: wait_for_picker_at_requested_shelf" in prompt
        assert "tag=wait_for_picker_here" in prompt
    finally:
        env.close()


def test_agv_unload_wait_prompt_says_hold_position_at_empty_shelf() -> None:
    env = make_env()
    try:
        agv = env.agents[0]
        picker = env.agents[1]
        delivered_shelf = env.shelfs[0]
        unload_action = action_id_for_coords(env, int(delivered_shelf.x), int(delivered_shelf.y))
        agv.carrying_shelf = delivered_shelf
        agv.has_delivered = True
        agv.target = unload_action
        agv.busy = True
        agv.x = int(delivered_shelf.x)
        agv.y = int(delivered_shelf.y)
        picker.target = unload_action
        picker.busy = True
        picker.x = int(delivered_shelf.x)
        picker.y = max(0, int(delivered_shelf.y) - 1)

        valid_masks = safe_valid_action_masks(env)
        prompt = build_agent_prompt(
            env,
            0,
            valid_masks,
            "language",
            10,
            persisted_actions_all=[unload_action, 0],
            hold_steps_remaining_all=[0, 0],
        )
        assert "waiting_for_picker_support_now: yes" in prompt
        assert "needs_picker_for_unload: yes" in prompt
        assert "next_required_flow_state: wait_for_picker_at_unload_shelf" in prompt
        assert "The AGV has already reached the EMPTY shelf return location." in prompt
        assert "Picker support is required here for unloading." in prompt
        assert "Hold position and wait for the Picker on this shelf." in prompt
        assert "picker_support_inbound_now: yes" in prompt
        assert "Picker is on the way to this same shelf now." in prompt
        assert "Do not leave this shelf now." in prompt
        assert "Preferred action family now: wait_at_unload_shelf" in prompt
        assert "tag=wait_for_picker_unload_here" in prompt
    finally:
        env.close()


def test_agv_prompt_omits_inbound_picker_instruction_when_picker_already_at_same_cell() -> None:
    env = make_env()
    try:
        agv = env.agents[0]
        picker = env.agents[1]
        requested_shelf = env.request_queue[0]
        target_action = action_id_for_coords(env, int(requested_shelf.x), int(requested_shelf.y))
        agv.target = target_action
        agv.busy = True
        agv.x = int(requested_shelf.x)
        agv.y = int(requested_shelf.y)
        picker.target = target_action
        picker.busy = True
        picker.x = int(requested_shelf.x)
        picker.y = int(requested_shelf.y)

        valid_masks = safe_valid_action_masks(env)
        prompt = build_agent_prompt(env, 0, valid_masks, "language", 10)
        assert "picker_at_same_cell: yes" in prompt
        assert "picker_support_inbound_now: no" in prompt
        assert "Picker is on the way to this same shelf now." not in prompt
        assert "Do not leave this shelf now." not in prompt
    finally:
        env.close()


def test_requested_shelves_context_is_trimmed_and_steps_guidance_is_structured() -> None:
    env = make_env()
    try:
        valid_masks = safe_valid_action_masks(env)
        prompt = build_agent_prompt(env, 0, valid_masks, "json", 10)
        full_payload = extract_prompt_json(prompt)
        payload = full_payload["planning_context"]
        assert len(payload["shared_coordination_context"]["requested_shelves"]) <= 5
        steps_guidance = payload["steps_guidance"]
        assert "movement_steps_hint" in steps_guidance
        assert "wait_steps_hint" in steps_guidance
        assert "charging_steps_hint" in steps_guidance
        first_candidate = payload["current_agent"]["candidate_actions"][0]
        assert "suggested_steps" in first_candidate
        assert "tag" in first_candidate
        compact_first_candidate = full_payload["candidate_actions"][0]
        assert "action" in compact_first_candidate
        assert "type" in compact_first_candidate
        assert "recommended_now" in compact_first_candidate
    finally:
        env.close()


def test_agv_candidates_show_primary_actions_before_charging_and_group_json_fields() -> None:
    env = make_env()
    try:
        valid_masks = safe_valid_action_masks(env)
        prompt = build_agent_prompt(env, 0, valid_masks, "language", 10)
        divider = "- IF CHARGING is needed, select following actions:"
        divider_index = prompt.index(divider)
        assert ":SHELF@" in prompt[:divider_index]
        assert ":CHARGING@" in prompt[divider_index:]

        payload = extract_prompt_json(build_agent_prompt(env, 0, valid_masks, "json", 10))["planning_context"]
        assert payload["current_agent"]["charging_candidate_header"] == "IF CHARGING is needed, select following actions:"
        assert payload["current_agent"]["primary_candidate_actions"]
        assert payload["current_agent"]["charging_candidate_actions"]
        assert all(
            row["action_type"] != "CHARGING"
            for row in payload["current_agent"]["primary_candidate_actions"]
        )
        assert all(
            row["action_type"] == "CHARGING"
            for row in payload["current_agent"]["charging_candidate_actions"]
        )
    finally:
        env.close()


def test_picker_prompt_and_json_use_same_charging_rule_and_grouped_candidates() -> None:
    env = make_env()
    try:
        valid_masks = safe_valid_action_masks(env)
        prompt = build_agent_prompt(env, 1, valid_masks, "language", 3)
        assert "If battery_need=not_needed and next_required_flow_state is not charge_now, charging is forbidden." in prompt
        divider = "- IF CHARGING is needed, select following actions:"
        divider_index = prompt.index(divider)
        assert ":CHARGING@" in prompt[divider_index:]

        payload = extract_prompt_json(build_agent_prompt(env, 1, valid_masks, "json", 3))["planning_context"]
        assert payload["current_agent"]["charging_candidate_header"] == "IF CHARGING is needed, select following actions:"
        assert all(
            row["action_type"] != "CHARGING"
            for row in payload["current_agent"]["primary_candidate_actions"]
        )
        assert all(
            row["action_type"] == "CHARGING"
            for row in payload["current_agent"]["charging_candidate_actions"]
        )
    finally:
        env.close()


def test_agv_prompt_advises_continued_charging_when_already_on_charger_below_threshold() -> None:
    env = make_env()
    try:
        agv = env.agents[0]
        charge_action = action_id_for_coords(env, 1, 0)
        unload_action = action_id_for_coords(env, 11, 4)
        agv.x = 1
        agv.y = 0
        agv.target = charge_action
        agv.busy = False
        agv.battery = 30.0
        agv.carrying_shelf = env.shelfs[0]
        env.request_queue = [shelf for shelf in env.request_queue if int(shelf.id) != int(agv.carrying_shelf.id)]
        valid_masks = safe_valid_action_masks(env)

        prompt = build_agent_prompt(
            env,
            0,
            valid_masks,
            "language",
            10,
            persisted_action=unload_action,
            remaining_hold_steps=0,
            persisted_actions_all=[unload_action, 0],
            hold_steps_remaining_all=[0, 0],
        )
        assert "next_required_flow_state: return_delivered_shelf_to_empty_shelf" in prompt
        assert "Once you are on a charging station with battery below 80, you must stay there until battery reaches at least 80 so you have enough charge for long tasks." in prompt
        assert "Ignore other requests and prioritize charging." in prompt
        assert "Preferred action family now: empty_shelf_return_or_charging" in prompt

        payload = extract_prompt_json(
            build_agent_prompt(
                env,
                0,
                valid_masks,
                "json",
                10,
                persisted_action=unload_action,
                remaining_hold_steps=0,
                persisted_actions_all=[unload_action, 0],
                hold_steps_remaining_all=[0, 0],
            )
        )["planning_context"]
        assert payload["current_agent"]["next_required_flow_state"] == "return_delivered_shelf_to_empty_shelf"
        assert payload["role_specific_context"]["agv_state"]["preferred_action_family"] == "empty_shelf_return_or_charging"
        assert payload["role_specific_context"]["charging_context"]["charge_until_task_ready_active"] is True
        assert payload["role_specific_context"]["charging_context"]["required_battery_for_release"] == 80
        assert payload["current_agent"]["charging_candidate_header"] == "IF CHARGING is needed, select following actions:"
    finally:
        env.close()


def test_agv_prompt_advises_continued_charging_when_picker_not_near_unload() -> None:
    env = make_env()
    try:
        agv = env.agents[0]
        charge_action = action_id_for_coords(env, 1, 0)
        unload_action = action_id_for_coords(env, 11, 4)
        agv.x = 1
        agv.y = 0
        agv.target = charge_action
        agv.busy = False
        agv.battery = 85.0
        agv.carrying_shelf = env.shelfs[0]
        env.request_queue = [shelf for shelf in env.request_queue if int(shelf.id) != int(agv.carrying_shelf.id)]
        valid_masks = safe_valid_action_masks(env)

        prompt = build_agent_prompt(
            env,
            0,
            valid_masks,
            "language",
            10,
            persisted_action=unload_action,
            remaining_hold_steps=0,
            persisted_actions_all=[unload_action, 0],
            hold_steps_remaining_all=[0, 0],
        )
        assert "next_required_flow_state: return_delivered_shelf_to_empty_shelf" in prompt
        assert "Once you are on this charging station, stay here until battery reaches at least" in prompt
        assert "Ignore other requests and prioritize charging while this threshold is not met." in prompt
        assert "Unload support actionable now: False." in prompt
        assert "Picker is not yet effectively in place for unload support, so charging should continue." in prompt
    finally:
        env.close()


def test_agv_prompt_may_resume_unload_flow_when_reserve_met_and_picker_near_unload() -> None:
    env = make_env()
    try:
        agv = env.agents[0]
        picker = env.agents[1]
        charge_action = action_id_for_coords(env, 1, 0)
        unload_action = action_id_for_coords(env, 11, 4)
        agv.x = 1
        agv.y = 0
        agv.target = charge_action
        agv.busy = False
        agv.battery = 85.0
        agv.carrying_shelf = env.shelfs[0]
        env.request_queue = [shelf for shelf in env.request_queue if int(shelf.id) != int(agv.carrying_shelf.id)]

        unload_coords = env.action_id_to_coords_map[unload_action]
        path = env.find_path((int(picker.y), int(picker.x)), unload_coords, picker, care_for_agents=False)
        assert path is not None and len(path) >= 3
        near_y, near_x = path[-3]
        picker.y = int(near_y)
        picker.x = int(near_x)
        picker.target = unload_action

        valid_masks = safe_valid_action_masks(env)
        assert next_required_flow_state(
            env,
            1,
            persisted_actions=[unload_action, unload_action],
            hold_steps_remaining=[0, 0],
        ) in {"move_to_support_unload", "hold_at_support_shelf"}

        prompt = build_agent_prompt(
            env,
            0,
            valid_masks,
            "language",
            10,
            persisted_action=unload_action,
            remaining_hold_steps=0,
            persisted_actions_all=[unload_action, unload_action],
            hold_steps_remaining_all=[0, 0],
        )
        assert "next_required_flow_state: return_delivered_shelf_to_empty_shelf" in prompt
        assert "Preferred action family now: empty_shelf_return_or_charging" not in prompt
        assert "Preferred action family now: empty_shelf_return" in prompt or "Preferred action family now: empty_shelf_return_or_charging" in prompt
        assert "Keep charging until battery reaches at least" not in prompt
    finally:
        env.close()


def test_candidate_actions_are_rendered_in_distance_order() -> None:
    env = make_env()
    try:
        valid_masks = safe_valid_action_masks(env)
        prompt = build_agent_prompt(env, 0, valid_masks, "language", 10)
        lines = [
            line.strip()
            for line in prompt.splitlines()
            if line.strip().startswith("- ")
            and "distance_steps=" in line
            and "IF CHARGING is needed, select following actions:" not in line
        ]
        candidate_lines = [line for line in lines if ":SHELF@" in line or ":CHARGING@" in line or ":GOAL@" in line or "0:NOOP" in line]
        primary_lines = [line for line in candidate_lines if ":CHARGING@" not in line]
        charging_lines = [line for line in candidate_lines if ":CHARGING@" in line]
        for line_group in (primary_lines, charging_lines):
            distances = []
            for line in line_group:
                match = re.search(r"distance_steps=(\d+)", line)
                if match is not None:
                    distances.append(int(match.group(1)))
            assert distances == sorted(distances)
    finally:
        env.close()


def test_noop_is_removed_when_productive_actions_exist() -> None:
    env = make_env()
    try:
        valid_masks = safe_valid_action_masks(env)
        agv_ids = obj2_candidate_action_ids(env, 0, valid_masks)
        picker_ids = obj2_candidate_action_ids(env, 1, valid_masks)
        assert 0 not in agv_ids
        assert 0 not in picker_ids
    finally:
        env.close()


def test_picker_noop_is_allowed_when_idle_and_battery_is_not_needed() -> None:
    env = make_env()
    try:
        picker = env.agents[1]
        picker.battery = 52.0

        agv = env.agents[0]
        agv.busy = False
        agv.target = 0
        agv.carrying_shelf = None

        valid_masks = safe_valid_action_masks(env)
        picker_ids = obj2_candidate_action_ids(env, 1, valid_masks)
        assert 0 in picker_ids

        prompt = build_agent_prompt(env, 1, valid_masks, "language", 3)
        assert "0:NOOP" in prompt
        assert "NOOP means stay idle / clear assignment" in prompt
    finally:
        env.close()


def test_order_agents_for_planning_uses_effective_support_state() -> None:
    env = make_env()
    try:
        requested_shelf = env.request_queue[0]
        target_action = action_id_for_coords(env, int(requested_shelf.x), int(requested_shelf.y))

        agv = env.agents[0]
        agv.x = int(requested_shelf.x)
        agv.y = int(requested_shelf.y)
        agv.target = 0
        agv.busy = False
        agv.carrying_shelf = None

        valid_masks = safe_valid_action_masks(env)
        ordering = order_agents_for_planning(env, valid_masks, [target_action, 0], [0, 0])
        assert ordering[0] == 0
        assert ordering[1] == 1
    finally:
        env.close()


def test_parse_steps_from_inline_text() -> None:
    text = "Reason: I choose Action 50:SHELF@(11,5) with Steps=3 because it is close."
    assert parse_steps_from_text(text, 10) == 3


def test_agv_empty_shelf_return_prompt_is_explicit() -> None:
    env = make_env()
    try:
        agv = env.agents[0]
        agv.carrying_shelf = env.shelfs[0]
        valid_masks = safe_valid_action_masks(env)

        prompt = build_agent_prompt(env, 0, valid_masks, "language", 10)
        assert "Empty shelf return task:" in prompt
        assert "Post-delivery helper:" in prompt
        assert "You already delivered shelf" in prompt
        assert "GOAL is finished for this shelf." in prompt
        assert "Do not return to the original request shelf just because the load started there." in prompt
        assert "The original request shelf is no longer the task after delivery." in prompt
        assert "Current unload shelf:" in prompt
        assert "Your carried shelf is already delivered." in prompt
        assert "GOAL is forbidden now." in prompt
        assert "Drop this shelf at one of these EMPTY shelf actions:" in prompt
        assert "Recommended EMPTY shelf action now:" in prompt
        assert "Your current task is the EMPTY shelf return location shown below." in prompt
        assert "This post-delivery phase is complete only after unload at that EMPTY shelf location." in prompt
        assert "This phase is complete only when carrying=None after unload." in prompt
        assert "tag=return_empty_shelf" in prompt
        assert "instruction=choose_this_to_unload_delivered_shelf" in prompt
        assert "recommended_now=yes" in prompt
        assert "Choose the candidate with recommended_now=yes to return and unload the delivered shelf." in prompt
        assert "If next_required_flow_state=return_delivered_shelf_to_empty_shelf, choose a tag=return_empty_shelf candidate." in prompt

        payload = extract_prompt_json(build_agent_prompt(env, 0, valid_masks, "json", 10))["planning_context"]
        assert payload["current_agent"]["next_required_flow_state"] == "return_delivered_shelf_to_empty_shelf"
        empty_ctx = payload["role_specific_context"]["empty_shelf_return_context"]
        assert empty_ctx["empty_shelf_return_required"] is True
        assert isinstance(empty_ctx["recommended_empty_shelf_actions"], list)
        assert empty_ctx["instruction"] == "choose one EMPTY shelf action, not GOAL"
        helper = payload["role_specific_context"]["post_delivery_helper"]
        assert helper["post_delivery_active"] is True
        assert helper["goal_forbidden_now"] is True
        assert helper["agv_next_task"] == "return_to_empty_shelf"
        assert "recommended_empty_shelf_action" in helper
        assert helper["current_post_delivery_unload_action_id"] == helper["recommended_empty_shelf_action_id"]
        assert "current_post_delivery_unload_position" in helper
        assert any(
            row["instruction"] == "choose_this_to_unload_delivered_shelf" and row["recommended_now"] is True
            for row in payload["current_agent"]["candidate_actions"]
        )
    finally:
        env.close()


def test_agv_prompt_tail_maps_flow_state_to_candidate_tags() -> None:
    env = make_env()
    try:
        agv = env.agents[0]
        requested_shelf = env.request_queue[0]
        agv.carrying_shelf = requested_shelf
        agv.busy = True
        agv.target = 1
        valid_masks = safe_valid_action_masks(env)

        prompt = build_agent_prompt(env, 0, valid_masks, "language", 10)
        assert "Follow next_required_flow_state first, then candidate tags, then suggested_steps." in prompt
        assert "If next_required_flow_state=move_requested_shelf_to_goal, choose a tag=deliver_now candidate." in prompt
        assert "tag=deliver_now" in prompt

        json_prompt = build_agent_prompt(env, 0, valid_masks, "json", 10)
        payload = extract_prompt_json(json_prompt)
        assert payload["current_control_state"]["next_required_flow_state"] == "move_requested_shelf_to_goal"
        assert any(row["tag"] == "deliver_now" and row["recommended_now"] is True for row in payload["candidate_actions"])
    finally:
        env.close()


def test_picker_prompt_tail_maps_support_flow_to_support_tag() -> None:
    env = make_env()
    try:
        agv = env.agents[0]
        requested_shelf = env.request_queue[0]
        target_action = action_id_for_coords(env, int(requested_shelf.x), int(requested_shelf.y))
        agv.target = target_action
        agv.busy = True
        valid_masks = safe_valid_action_masks(env)

        prompt = build_agent_prompt(env, 1, valid_masks, "language", 3)
        assert "If next_required_flow_state=move_to_support_load or move_to_support_unload, choose a tag=support_now candidate." in prompt
        assert "If support_needed_now=yes and battery is not critical, choose the matching support shelf action." in prompt
        assert "tag=support_now" in prompt

        json_prompt = build_agent_prompt(env, 1, valid_masks, "json", 3)
        payload = extract_prompt_json(json_prompt)
        assert payload["current_control_state"]["next_required_flow_state"] == "move_to_support_load"
        assert any(row["tag"] == "support_now" for row in payload["candidate_actions"])
    finally:
        env.close()


def test_picker_post_load_helper_is_added_when_agv_has_loaded_requested_shelf() -> None:
    env = make_env()
    try:
        agv = env.agents[0]
        requested_shelf = env.request_queue[0]
        agv.carrying_shelf = requested_shelf
        agv.busy = True
        agv.target = 1

        valid_masks = safe_valid_action_masks(env)
        prompt = build_agent_prompt(env, 1, valid_masks, "language", 3)
        assert "Post-load helper:" in prompt
        assert f"AGV agent_0 has already loaded the requested shelf." in prompt
        assert "The load support phase at the original shelf is complete." in prompt
        assert "Do not keep selecting the old load shelf unless another AGV needs support there." in prompt
        assert "The AGV should now move to GOAL." in prompt
        assert "Your support priority should shift away from that completed load shelf." in prompt

        payload = extract_prompt_json(build_agent_prompt(env, 1, valid_masks, "json", 3))["planning_context"]
        helper = payload["role_specific_context"]["post_load_helper"]
        assert helper["post_load_active"] is True
        assert helper["agv_id_loaded_requested_shelf"] == 0
    finally:
        env.close()


def test_picker_post_delivery_helper_is_added_for_unload_support() -> None:
    env = make_env()
    try:
        agv = env.agents[0]
        agv.carrying_shelf = env.shelfs[0]
        unload_action = action_id_for_coords(env, 2, 2)
        agv.target = unload_action
        agv.busy = True

        valid_masks = safe_valid_action_masks(env)
        prompt = build_agent_prompt(env, 1, valid_masks, "language", 3)
        assert "Post-delivery helper:" in prompt
        assert "already delivered its shelf and is still carrying it." in prompt
        assert "post-delivery unload support phase" in prompt
        assert "Picker support candidate shelf actions:" in prompt
        assert f"{unload_action}:SHELF" in prompt
        assert "support_need=moving_to_shelf_will_need_picker_for_unload" in prompt
        assert "state_source=post_delivery_helper" in prompt
        assert "Do not wait at the old request shelf after delivery." in prompt
        assert "The current support location is the EMPTY shelf return location, not the original request shelf." in prompt
        assert "Support the AGV at the same unload shelf action id." in prompt
        assert "Current unload shelf:" in prompt
        assert "Recommended unload support action now:" in prompt
        assert "Move to the same unload shelf action id as the AGV." in prompt
        assert "Do not choose charging unless battery is critical." in prompt
        assert "This post-delivery phase is complete only after the AGV unloads the shelf there." in prompt
        assert "instruction=choose_this_to_support_unloading_on_exact_same_shelf_cell" in prompt
        assert "recommended_now=yes" in prompt
        assert "Choose the candidate with recommended_now=yes to support unloading on the exact same shelf cell." in prompt

        payload = extract_prompt_json(build_agent_prompt(env, 1, valid_masks, "json", 3))["planning_context"]
        helper = payload["role_specific_context"]["post_delivery_helper"]
        support_context = payload["role_specific_context"]["picker_support_context"]
        support_rows = payload["shared_coordination_context"]["agvs_requesting_picker_support"]
        assert helper["post_delivery_active"] is True
        assert helper["support_type"] == "unload"
        assert helper["support_action_id"] == unload_action
        assert helper["current_post_delivery_unload_action_id"] == unload_action
        assert helper["current_post_delivery_unload_position"] == helper["support_position"]
        assert support_context["support_target_action_id"] == unload_action
        assert any(f"target={unload_action}:SHELF" in row for row in support_rows)
        assert any(
            row["instruction"] == "choose_this_to_support_unloading_on_exact_same_shelf_cell" and row["recommended_now"] is True
            for row in payload["current_agent"]["candidate_actions"]
        )
    finally:
        env.close()


def test_post_delivery_helper_is_inactive_when_not_needed() -> None:
    env = make_env()
    try:
        valid_masks = safe_valid_action_masks(env)
        prompt = build_agent_prompt(env, 0, valid_masks, "language", 10)
        assert "Post-delivery helper:" not in prompt

        payload = extract_prompt_json(build_agent_prompt(env, 0, valid_masks, "json", 10))["planning_context"]
        helper = payload["role_specific_context"]["post_delivery_helper"]
        assert helper["post_delivery_active"] is False
    finally:
        env.close()


def test_picker_post_delivery_helper_and_support_candidates_never_disagree() -> None:
    env = make_env()
    try:
        agv = env.agents[0]
        agv.carrying_shelf = env.shelfs[0]
        agv.target = 0
        agv.busy = False

        valid_masks = safe_valid_action_masks(env)
        prompt = build_agent_prompt(env, 1, valid_masks, "language", 3)
        assert "Post-delivery helper:" in prompt
        helper_match = re.search(r"Recommended unload support action now: (\d+):SHELF", prompt)
        assert helper_match is not None
        helper_action = int(helper_match.group(1))
        assert f"{helper_action}:SHELF" in prompt
        assert "Picker support candidate shelf actions:\n- none" not in prompt

        payload = extract_prompt_json(build_agent_prompt(env, 1, valid_masks, "json", 3))["planning_context"]
        helper = payload["role_specific_context"]["post_delivery_helper"]
        picker_context = payload["role_specific_context"]["picker_support_context"]
        support_rows = payload["shared_coordination_context"]["agvs_requesting_picker_support"]
        assert helper["post_delivery_active"] is True
        assert helper["support_action_id"] == helper_action
        assert picker_context["support_target_action_id"] == helper_action
        assert any(f"target={helper_action}:SHELF" in row for row in support_rows)
    finally:
        env.close()


def test_environment_info_exposes_per_agent_stuck_and_conflict_fields() -> None:
    env = make_env()
    try:
        valid_masks = safe_valid_action_masks(env)
        actions = [
            obj2_candidate_action_ids(env, 0, valid_masks)[0],
            obj2_candidate_action_ids(env, 1, valid_masks)[0],
        ]
        _, _, _, _, info = env.step(actions)
        assert "agent_stuck_counts" in info
        assert "stuck_agents" in info
        assert "agents_in_conflict" in info
        assert "agent_resolving_conflict" in info
        assert "agent_conflict_resolution_steps" in info
        assert "agent_resolution_failed_recently" in info
        assert "agent_unreachable_target_action_id" in info
        assert "agent_unreachable_target_reason" in info
        assert "agent_unreachable_target_cooldown_steps" in info
        assert "agent_req_actions" in info
        assert "agent_targets" in info
        assert len(info["agent_stuck_counts"]) == env.num_agents
        assert len(info["stuck_agents"]) == env.num_agents
        assert len(info["agents_in_conflict"]) == env.num_agents
        assert len(info["agent_resolving_conflict"]) == env.num_agents
        assert len(info["agent_conflict_resolution_steps"]) == env.num_agents
        assert len(info["agent_resolution_failed_recently"]) == env.num_agents
        assert len(info["agent_unreachable_target_action_id"]) == env.num_agents
        assert len(info["agent_unreachable_target_cooldown_steps"]) == env.num_agents
    finally:
        env.close()


def test_plan_step_skips_llm_query_when_env_owned_conflict_resolution_is_active() -> None:
    env = make_env()
    try:
        valid_masks = safe_valid_action_masks(env)
        persisted_actions = [action_id_for_coords(env, int(env.request_queue[0].x), int(env.request_queue[0].y)), 0]
        hold_steps_remaining = [0, 0]
        env.agents[0].busy = True
        env.agents[0].target = persisted_actions[0]

        class Args:
            max_action_hold_steps = 10
            max_picker_hold_steps = 3
            prompt_format = "language"
            ollama_url = "http://unused"
            request_timeout_s = 1
            temperature = 0.0
            num_predict = 32

        actions, metrics, query_records = plan_step_sequential(
            env,
            valid_masks,
            Args(),
            "unused-model",
            persisted_actions,
            hold_steps_remaining,
            latest_env_info={
                "agent_resolving_conflict": [True, False],
                "agent_conflict_resolution_steps": [3, 0],
                "agent_targets": [persisted_actions[0], 0],
                "agent_req_actions": ["NOOP", "NOOP"],
            },
        )
        assert actions[0] == persisted_actions[0]
        assert metrics["env_conflict_resolution_skips_this_step"] == 1
        agv_record = next(row for row in query_records if row["agent_id"] == 0)
        assert agv_record["resolution"] == "env_owned_conflict_resolution_without_llm_query"
        assert agv_record["env_conflict_resolution_active"] is True
        assert agv_record["agent_conflict_resolution_steps"] == 3
        assert agv_record["suppressed_llm_query_reason"] == "env_owned_conflict_resolution_active"
    finally:
        env.close()


def test_order_agents_for_planning_keeps_busy_agent_skipped_below_override_threshold() -> None:
    env = make_env()
    try:
        env.agents[0].busy = True
        ordering = order_agents_for_planning(
            env,
            safe_valid_action_masks(env),
            busy_steps_by_agent=[2, 0],
            max_busy_steps_for_replan=3,
            latest_env_info={"agent_resolving_conflict": [False, False]},
        )
        assert 0 not in ordering
    finally:
        env.close()


def test_order_agents_for_planning_includes_busy_shelf_agent_at_override_threshold() -> None:
    env = make_env()
    try:
        agv = env.agents[0]
        requested_shelf = env.request_queue[0]
        agv.busy = True
        agv.charging = False
        agv.target = action_id_for_coords(env, int(requested_shelf.x), int(requested_shelf.y))
        ordering = order_agents_for_planning(
            env,
            safe_valid_action_masks(env),
            busy_steps_by_agent=[3, 0],
            max_busy_steps_for_replan=3,
            latest_env_info={"agent_resolving_conflict": [False, False]},
        )
        assert 0 in ordering
    finally:
        env.close()


def test_order_agents_for_planning_does_not_interrupt_busy_charging_agent_at_threshold() -> None:
    env = make_env()
    try:
        agv = env.agents[0]
        charge_action = action_id_for_coords(env, int(env.charging_stations[0].x), int(env.charging_stations[0].y))
        agv.busy = True
        agv.charging = True
        agv.target = charge_action
        ordering = order_agents_for_planning(
            env,
            safe_valid_action_masks(env),
            busy_steps_by_agent=[5, 0],
            max_busy_steps_for_replan=3,
            latest_env_info={"agent_resolving_conflict": [False, False]},
        )
        assert 0 not in ordering
    finally:
        env.close()


def test_order_agents_for_planning_does_not_interrupt_conflict_resolution_at_threshold() -> None:
    env = make_env()
    try:
        env.agents[0].busy = True
        ordering = order_agents_for_planning(
            env,
            safe_valid_action_masks(env),
            busy_steps_by_agent=[5, 0],
            max_busy_steps_for_replan=3,
            latest_env_info={"agent_resolving_conflict": [True, False]},
        )
        assert 0 not in ordering
    finally:
        env.close()


def test_order_agents_for_planning_does_not_interrupt_agv_waiting_at_unload_shelf() -> None:
    env = make_env()
    try:
        agv = env.agents[0]
        delivered_shelf = env.shelfs[0]
        unload_action = action_id_for_coords(env, int(delivered_shelf.x), int(delivered_shelf.y))
        agv.carrying_shelf = delivered_shelf
        agv.has_delivered = True
        agv.busy = True
        agv.target = unload_action
        agv.x = int(delivered_shelf.x)
        agv.y = int(delivered_shelf.y)
        env._recalc_grid()

        assert next_required_flow_state(env, 0, [unload_action, 0], [0, 0]) == "wait_for_picker_at_unload_shelf"
        assert agent_is_interruptible_when_busy(
            env,
            0,
            latest_env_info={"agent_resolving_conflict": [False, False]},
        ) is False

        ordering = order_agents_for_planning(
            env,
            safe_valid_action_masks(env),
            persisted_actions=[unload_action, 0],
            hold_steps_remaining=[0, 0],
            busy_steps_by_agent=[5, 0],
            max_busy_steps_for_replan=3,
            latest_env_info={"agent_resolving_conflict": [False, False]},
        )
        assert 0 not in ordering
    finally:
        env.close()


def test_order_agents_for_planning_protects_agv_unload_wait_for_first_five_steps_even_if_not_busy() -> None:
    env = make_env()
    try:
        agv = env.agents[0]
        picker = env.agents[1]
        delivered_shelf = env.shelfs[0]
        unload_action = action_id_for_coords(env, int(delivered_shelf.x), int(delivered_shelf.y))
        agv.carrying_shelf = delivered_shelf
        agv.has_delivered = True
        agv.busy = False
        agv.target = unload_action
        agv.x = int(delivered_shelf.x)
        agv.y = int(delivered_shelf.y)
        picker.x = int(delivered_shelf.x) + 2
        picker.y = int(delivered_shelf.y)
        env._recalc_grid()

        assert agv_requires_protected_unload_wait(env, 0, [unload_action, 0], [0, 0]) is True
        ordering = order_agents_for_planning(
            env,
            safe_valid_action_masks(env),
            persisted_actions=[unload_action, 0],
            hold_steps_remaining=[0, 0],
            unload_wait_steps_by_agent=[3, 0],
            max_busy_steps_for_replan=3,
            latest_env_info={"agent_resolving_conflict": [False, False]},
        )
        assert 0 not in ordering
    finally:
        env.close()


def test_plan_step_sequential_holds_agv_at_unload_shelf_during_protected_wait_even_if_battery_is_critical() -> None:
    env = make_env()
    try:
        agv = env.agents[0]
        picker = env.agents[1]
        delivered_shelf = env.shelfs[0]
        unload_action = action_id_for_coords(env, int(delivered_shelf.x), int(delivered_shelf.y))
        agv.carrying_shelf = delivered_shelf
        agv.has_delivered = True
        agv.busy = False
        agv.battery = 0.0
        agv.target = unload_action
        agv.x = int(delivered_shelf.x)
        agv.y = int(delivered_shelf.y)
        picker.busy = True
        picker.target = action_id_for_coords(env, int(env.charging_stations[0].x), int(env.charging_stations[0].y))
        env._recalc_grid()
        valid_masks = safe_valid_action_masks(env)

        class Args:
            max_action_hold_steps = 10
            max_picker_hold_steps = 3
            prompt_format = "language"
            ollama_url = "http://unused"
            request_timeout_s = 1
            temperature = 0.0
            num_predict = 32
            max_busy_steps_for_replan = 3

        actions, metrics, query_records = plan_step_sequential(
            env,
            valid_masks,
            Args(),
            "unused-model",
            persisted_actions=[unload_action, picker.target],
            hold_steps_remaining=[0, 0],
            busy_steps_by_agent=[0, 1],
            unload_wait_steps_by_agent=[3, 0],
            latest_env_info={
                "agent_resolving_conflict": [False, False],
                "agent_targets": [unload_action, int(picker.target)],
                "agent_req_actions": ["TOGGLE_LOAD", "NOOP"],
                "agent_resolution_failed_recently": [False, False],
                "agent_unreachable_target_action_id": [0, 0],
                "agent_unreachable_target_reason": [None, None],
                "agent_unreachable_target_cooldown_steps": [0, 0],
            },
        )
        assert actions[0] == unload_action
        assert metrics["llm_calls_this_step"] == 0
        assert all(row["agent_id"] != 0 for row in query_records)
    finally:
        env.close()


def test_plan_step_sequential_keeps_agv_on_unload_shelf_for_post_arrival_retry() -> None:
    env = make_env()
    try:
        agv = env.agents[0]
        picker = env.agents[1]
        delivered_shelf = env.shelfs[0]
        unload_action = action_id_for_coords(env, int(delivered_shelf.x), int(delivered_shelf.y))
        agv.carrying_shelf = delivered_shelf
        agv.has_delivered = True
        agv.busy = False
        agv.target = unload_action
        agv.x = int(delivered_shelf.x)
        agv.y = int(delivered_shelf.y)
        picker.x = int(delivered_shelf.x)
        picker.y = int(delivered_shelf.y)
        picker.busy = False
        picker.target = unload_action
        env._recalc_grid()
        valid_masks = safe_valid_action_masks(env)

        class Args:
            max_action_hold_steps = 10
            max_picker_hold_steps = 3
            prompt_format = "language"
            ollama_url = "http://unused"
            request_timeout_s = 1
            temperature = 0.0
            num_predict = 32
            max_busy_steps_for_replan = 3

        actions, metrics, query_records = plan_step_sequential(
            env,
            valid_masks,
            Args(),
            "unused-model",
            persisted_actions=[unload_action, unload_action],
            hold_steps_remaining=[0, 0],
            busy_steps_by_agent=[0, 0],
            unload_wait_steps_by_agent=[3, 0],
            latest_env_info={
                "agent_resolving_conflict": [False, False],
                "agent_targets": [unload_action, unload_action],
                "agent_req_actions": ["TOGGLE_LOAD", "NOOP"],
                "agent_resolution_failed_recently": [False, False],
                "agent_unreachable_target_action_id": [0, 0],
                "agent_unreachable_target_reason": [None, None],
                "agent_unreachable_target_cooldown_steps": [0, 0],
            },
        )
        assert actions[0] == unload_action
        assert metrics["llm_calls_this_step"] == 0
        assert all(row["agent_id"] != 0 for row in query_records)
    finally:
        env.close()


def test_same_target_reassignment_is_noop_while_resolving_conflict() -> None:
    env = make_env()
    try:
        requested_shelf = env.request_queue[0]
        target_action = action_id_for_coords(env, int(requested_shelf.x), int(requested_shelf.y))
        charging_cells = {(int(station.x), int(station.y)) for station in env.charging_stations}
        agv = env.agents[0]

        env._assign_macro_target(agv, target_action, charging_cells)
        original_path = list(agv.path)
        agv.resolving_conflict = True
        agv.conflict_resolution_steps = 2
        agv.resolution_target = target_action

        env._assign_macro_target(agv, target_action, charging_cells)

        assert agv.target == target_action
        assert agv.busy is True
        assert agv.path == original_path
        assert agv.resolving_conflict is True
        assert agv.conflict_resolution_steps == 2
    finally:
        env.close()


def test_successful_load_clears_completed_shelf_assignment() -> None:
    env = make_env()
    try:
        requested_shelf = env.request_queue[0]
        agv = env.agents[0]
        picker = env.agents[1]
        agv.x = int(requested_shelf.x)
        agv.y = int(requested_shelf.y)
        picker.x = int(requested_shelf.x)
        picker.y = int(requested_shelf.y)
        agv.target = action_id_for_coords(env, int(requested_shelf.x), int(requested_shelf.y))
        agv.busy = True
        agv.req_action = Action.TOGGLE_LOAD
        env._recalc_grid()

        rewards = np.zeros(env.num_agents)
        rewards = env.execute_micro_actions(rewards)

        assert agv.carrying_shelf is not None
        assert agv.busy is False
        assert agv.target == 0
        assert agv.path == []
        assert agv.resolving_conflict is False
    finally:
        env.close()


def test_successful_unload_clears_completed_shelf_assignment() -> None:
    env = make_env()
    try:
        requested_shelf = env.request_queue[0]
        agv = env.agents[0]
        picker = env.agents[1]
        agv.x = int(requested_shelf.x)
        agv.y = int(requested_shelf.y)
        picker.x = int(requested_shelf.x)
        picker.y = int(requested_shelf.y)
        agv.carrying_shelf = requested_shelf
        agv.target = action_id_for_coords(env, int(requested_shelf.x), int(requested_shelf.y))
        agv.busy = True
        agv.req_action = Action.TOGGLE_LOAD
        env._recalc_grid()

        rewards = np.zeros(env.num_agents)
        rewards = env.execute_micro_actions(rewards)

        assert agv.carrying_shelf is None
        assert agv.busy is False
        assert agv.target == 0
        assert agv.path == []
        assert agv.resolving_conflict is False
    finally:
        env.close()


def test_conflict_timeout_marks_target_temporarily_unreachable() -> None:
    env = make_env()
    try:
        requested_shelf = env.request_queue[0]
        target_action = action_id_for_coords(env, int(requested_shelf.x), int(requested_shelf.y))
        agv = env.agents[0]
        agv.busy = True
        agv.target = target_action
        agv.resolving_conflict = True
        agv.conflict_resolution_steps = 15
        agv.resolution_target = target_action

        env.attribute_macro_actions([target_action, 0])

        assert agv.busy is False
        assert agv.target == 0
        assert agv.resolving_conflict is False
        assert agv.resolution_failed_recently is True
        assert agv.unreachable_target_action_id == target_action
        assert agv.unreachable_target_reason == "conflict_timeout_target_not_reachable"
        assert agv.unreachable_target_cooldown_steps == 3
    finally:
        env.close()


def test_unreachable_target_cooldown_decays_and_clears() -> None:
    env = make_env()
    try:
        agv = env.agents[0]
        agv.resolution_failed_recently = True
        agv.unreachable_target_action_id = 50
        agv.unreachable_target_reason = "conflict_timeout_target_not_reachable"
        agv.unreachable_target_cooldown_steps = 1

        env.attribute_macro_actions([0, 0])

        assert agv.resolution_failed_recently is False
        assert agv.unreachable_target_action_id == 0
        assert agv.unreachable_target_reason is None
        assert agv.unreachable_target_cooldown_steps == 0
    finally:
        env.close()


def test_plan_step_replans_after_env_marks_previous_target_unreachable() -> None:
    env = make_env()
    try:
        valid_masks = safe_valid_action_masks(env)
        failed_target = action_id_for_coords(env, int(env.request_queue[0].x), int(env.request_queue[0].y))
        charging_action = action_id_for_coords(
            env,
            int(env.charging_stations[0].x),
            int(env.charging_stations[0].y),
        )
        persisted_actions = [failed_target, 0]
        hold_steps_remaining = [2, 0]

        class Args:
            max_action_hold_steps = 10
            max_picker_hold_steps = 3
            prompt_format = "language"
            ollama_url = "http://unused"
            request_timeout_s = 1
            temperature = 0.0
            num_predict = 32

        with patch(
            "scripts.run_obj2_shared_context_llm.query_ollama_text",
            return_value=f"Action: {charging_action}\nSteps: 1",
        ) as mocked_query:
            actions, metrics, query_records = plan_step_sequential(
                env,
                valid_masks,
                Args(),
                "unused-model",
                persisted_actions,
                hold_steps_remaining,
                latest_env_info={
                    "agent_resolving_conflict": [False, False],
                    "agent_conflict_resolution_steps": [0, 0],
                    "agent_targets": [0, 0],
                    "agent_req_actions": ["NOOP", "NOOP"],
                    "agent_resolution_failed_recently": [True, False],
                    "agent_unreachable_target_action_id": [failed_target, 0],
                    "agent_unreachable_target_reason": ["conflict_timeout_target_not_reachable", None],
                    "agent_unreachable_target_cooldown_steps": [3, 0],
                },
            )

        assert mocked_query.call_count == 2
        assert actions[0] == charging_action
        assert persisted_actions[0] == charging_action
        assert hold_steps_remaining[0] == 0
        agv_record = next(row for row in query_records if row["agent_id"] == 0)
        assert agv_record["resolution"] == "accepted_llm_action"
        assert agv_record["resolution_failed_recently"] is True
        assert agv_record["unreachable_target_action_id"] == failed_target
        assert metrics["action_reuse_this_step"] == 0
    finally:
        env.close()


def test_plan_step_replans_busy_agent_after_max_busy_steps_and_clears_stale_hold() -> None:
    env = make_env()
    try:
        valid_masks = safe_valid_action_masks(env)
        shelf_action = action_id_for_coords(env, int(env.request_queue[0].x), int(env.request_queue[0].y))
        charging_action = action_id_for_coords(env, int(env.charging_stations[0].x), int(env.charging_stations[0].y))
        persisted_actions = [shelf_action, 0]
        hold_steps_remaining = [4, 0]
        busy_steps_by_agent = [5, 0]
        env.agents[0].busy = True
        env.agents[0].charging = False
        env.agents[0].target = shelf_action

        class Args:
            max_action_hold_steps = 10
            max_picker_hold_steps = 3
            max_busy_steps_for_replan = 5
            prompt_format = "language"
            ollama_url = "http://unused"
            request_timeout_s = 1
            temperature = 0.0
            num_predict = 32

        with patch(
            "scripts.run_obj2_shared_context_llm.query_ollama_text",
            return_value=f"Action: {charging_action}\nSteps: 1",
        ) as mocked_query:
            actions, metrics, query_records = plan_step_sequential(
                env,
                valid_masks,
                Args(),
                "unused-model",
                persisted_actions,
                hold_steps_remaining,
                busy_steps_by_agent,
                latest_env_info={
                    "agent_resolving_conflict": [False, False],
                    "agent_conflict_resolution_steps": [0, 0],
                    "agent_targets": [shelf_action, 0],
                    "agent_req_actions": ["NOOP", "NOOP"],
                },
            )

        assert mocked_query.call_count == 2
        assert actions[0] == charging_action
        assert persisted_actions[0] == charging_action
        assert hold_steps_remaining[0] == 0
        agv_record = next(row for row in query_records if row["agent_id"] == 0)
        assert agv_record["resolution"] == "busy_override_replanned_after_max_busy_steps"
        assert agv_record["busy_override_active"] is True
        assert agv_record["busy_steps_count"] == 5
        assert agv_record["busy_override_threshold"] == 5
        assert metrics["action_reuse_this_step"] == 0
    finally:
        env.close()


def test_run_single_episode_records_max_busy_steps_for_replan_in_summary(tmp_path) -> None:
    class DummyArgs:
        seed = 0
        max_steps = 1
        max_inactivity_steps = 200
        max_busy_steps_for_replan = 7
        prompt_format = "language"
        render = False

    class DummyEnvWrapper:
        def __init__(self):
            self.unwrapped = make_env()

        def reset(self, seed=None):
            return self.unwrapped.reset(seed=seed)

        def step(self, actions):
            obs, rewards, terminateds, truncateds, info = self.unwrapped.step(actions)
            return obs, rewards, [True for _ in terminateds], [True for _ in truncateds], info

        def close(self):
            self.unwrapped.close()

    scenario = ScenarioSpec(
        env_id="tarware-tiny-1agvs-1pickers-globalobs-chg-v1",
        label="tiny_balanced_1v1",
        focus="test",
    )
    with patch("scripts.run_obj2_shared_context_llm.gym.make", return_value=DummyEnvWrapper()):
        summary = run_single_episode(DummyArgs(), "dummy-model", "SLM", scenario, tmp_path)

    assert summary["max_busy_steps_for_replan"] == 7


def test_persisted_action_completed_by_env_after_unload_completion() -> None:
    env = make_env()
    try:
        requested_shelf = env.request_queue[0]
        unload_action = action_id_for_coords(env, int(requested_shelf.x), int(requested_shelf.y))
        agv = env.agents[0]
        agv.x = int(requested_shelf.x)
        agv.y = int(requested_shelf.y)
        agv.busy = False
        agv.target = 0
        agv.charging = False
        agv.carrying_shelf = None
        env._recalc_grid()

        assert persisted_action_completed_by_env(env, 0, unload_action, 5, {}) is True
    finally:
        env.close()


def test_persisted_action_completed_by_env_after_load_completion() -> None:
    env = make_env()
    try:
        requested_shelf = env.request_queue[0]
        shelf_action = action_id_for_coords(env, int(requested_shelf.x), int(requested_shelf.y))
        agv = env.agents[0]
        agv.x = int(requested_shelf.x)
        agv.y = int(requested_shelf.y)
        agv.busy = False
        agv.target = 0
        agv.charging = False
        agv.carrying_shelf = requested_shelf
        env._recalc_grid()

        assert persisted_action_completed_by_env(env, 0, shelf_action, 5, {}) is True
    finally:
        env.close()


def test_persisted_action_not_completed_during_valid_agv_wait_at_shelf() -> None:
    env = make_env()
    try:
        requested_shelf = env.request_queue[0]
        shelf_action = action_id_for_coords(env, int(requested_shelf.x), int(requested_shelf.y))
        agv = env.agents[0]
        agv.x = int(requested_shelf.x)
        agv.y = int(requested_shelf.y)
        agv.busy = True
        agv.target = shelf_action
        agv.charging = False
        agv.carrying_shelf = None
        env._recalc_grid()

        assert persisted_action_completed_by_env(env, 0, shelf_action, 5, {}) is False
    finally:
        env.close()


def test_persisted_action_not_completed_during_valid_picker_wait_at_support_shelf() -> None:
    env = make_env()
    try:
        requested_shelf = env.request_queue[0]
        shelf_action = action_id_for_coords(env, int(requested_shelf.x), int(requested_shelf.y))
        picker = env.agents[1]
        picker.x = int(requested_shelf.x)
        picker.y = int(requested_shelf.y)
        picker.busy = True
        picker.target = shelf_action
        picker.charging = False
        env._recalc_grid()

        assert persisted_action_completed_by_env(env, 1, shelf_action, 5, {}) is False
    finally:
        env.close()


def test_persisted_action_not_completed_while_charging_hold_remains_and_battery_not_full() -> None:
    env = make_env()
    try:
        charge_action = action_id_for_coords(env, int(env.charging_stations[0].x), int(env.charging_stations[0].y))
        picker = env.agents[1]
        picker.x = int(env.charging_stations[0].x)
        picker.y = int(env.charging_stations[0].y)
        picker.charging = True
        picker.busy = False
        picker.target = charge_action
        picker.battery = 55.0
        env._recalc_grid()

        assert persisted_action_completed_by_env(env, 1, charge_action, 2, {}) is False
    finally:
        env.close()


def test_persisted_action_not_completed_when_charging_hold_is_zero_but_battery_not_full() -> None:
    env = make_env()
    try:
        charge_action = action_id_for_coords(env, int(env.charging_stations[0].x), int(env.charging_stations[0].y))
        picker = env.agents[1]
        picker.x = int(env.charging_stations[0].x)
        picker.y = int(env.charging_stations[0].y)
        picker.charging = True
        picker.busy = False
        picker.target = charge_action
        picker.battery = 55.0
        env._recalc_grid()

        assert persisted_action_completed_by_env(env, 1, charge_action, 0, {}) is False
    finally:
        env.close()


def test_persisted_action_completed_when_charging_finishes() -> None:
    env = make_env()
    try:
        charge_action = action_id_for_coords(env, int(env.charging_stations[0].x), int(env.charging_stations[0].y))
        picker = env.agents[1]
        picker.x = int(env.charging_stations[0].x)
        picker.y = int(env.charging_stations[0].y)
        picker.charging = True
        picker.busy = False
        picker.target = charge_action
        picker.battery = 100.0
        env._recalc_grid()

        assert persisted_action_completed_by_env(env, 1, charge_action, 2, {}) is True
    finally:
        env.close()


def test_plan_step_clears_stale_hold_after_completed_unload() -> None:
    env = make_env()
    try:
        valid_masks = safe_valid_action_masks(env)
        requested_shelf = env.request_queue[0]
        shelf_action = action_id_for_coords(env, int(requested_shelf.x), int(requested_shelf.y))
        agv = env.agents[0]
        agv.x = int(requested_shelf.x)
        agv.y = int(requested_shelf.y)
        agv.busy = False
        agv.target = 0
        agv.charging = False
        agv.carrying_shelf = None
        env._recalc_grid()

        persisted_actions = [shelf_action, 0]
        hold_steps_remaining = [5, 0]

        class Args:
            max_action_hold_steps = 10
            max_picker_hold_steps = 3
            prompt_format = "language"
            ollama_url = "http://unused"
            request_timeout_s = 1
            temperature = 0.0
            num_predict = 32

        with patch(
            "scripts.run_obj2_shared_context_llm.query_ollama_text",
            return_value="Action: 55\nSteps: 1",
        ) as mocked_query:
            actions, metrics, query_records = plan_step_sequential(
                env,
                valid_masks,
                Args(),
                "unused-model",
                persisted_actions,
                hold_steps_remaining,
                latest_env_info={},
            )

        assert mocked_query.call_count == 2
        assert actions[0] != shelf_action
        assert persisted_actions[0] != shelf_action
        agv_record = next(row for row in query_records if row["agent_id"] == 0)
        assert agv_record["resolution"] == "accepted_llm_action"
    finally:
        env.close()


def test_env_emits_charge_on_station_even_when_busy_is_false() -> None:
    env = make_env()
    try:
        agv = env.agents[0]
        charge_action = action_id_for_coords(env, int(env.charging_stations[0].x), int(env.charging_stations[0].y))
        agv.x = int(env.charging_stations[0].x)
        agv.y = int(env.charging_stations[0].y)
        agv.target = charge_action
        agv.charging = True
        agv.busy = False
        agv.battery = 20.0
        env._recalc_grid()

        env.attribute_macro_actions([charge_action, 0])

        assert agv.req_action == Action.CHARGE
    finally:
        env.close()


def test_env_repeated_charge_steps_increase_battery_until_full() -> None:
    env = make_env()
    try:
        agv = env.agents[0]
        charge_action = action_id_for_coords(env, int(env.charging_stations[0].x), int(env.charging_stations[0].y))
        agv.x = int(env.charging_stations[0].x)
        agv.y = int(env.charging_stations[0].y)
        agv.target = charge_action
        agv.charging = True
        agv.busy = False
        agv.battery = 90.0
        env._recalc_grid()

        rewards = np.zeros(env.num_agents)
        env.attribute_macro_actions([charge_action, 0])
        rewards = env.execute_micro_actions(rewards)
        assert agv.battery == 95.0

        env.attribute_macro_actions([charge_action, 0])
        rewards = env.execute_micro_actions(rewards)
        assert agv.battery == 100.0

        env.attribute_macro_actions([charge_action, 0])
        assert agv.req_action == Action.NOOP
        assert agv.target == 0
    finally:
        env.close()


def test_language_prompt_warns_when_previous_target_is_temporarily_unreachable() -> None:
    env = make_env()
    try:
        agv = env.agents[0]
        failed_target = action_id_for_coords(env, int(env.request_queue[0].x), int(env.request_queue[0].y))
        agv.resolution_failed_recently = True
        agv.unreachable_target_action_id = failed_target
        agv.unreachable_target_reason = "conflict_timeout_target_not_reachable"
        agv.unreachable_target_cooldown_steps = 3
        valid_masks = safe_valid_action_masks(env)

        prompt = build_agent_prompt(env, 0, valid_masks, "language", 10)

        assert "Execution warning:" in prompt
        assert f"The current agent is stuck on target action {failed_target}." in prompt
        assert f"Target action {failed_target} is temporarily unreachable right now." in prompt
        assert f"Do not select action {failed_target} in this response." in prompt
        assert "You must choose a different valid action from Candidate actions." in prompt
        assert "selecting that same action id is invalid for this turn; choose another valid candidate." in prompt
        assert "tag=temporarily_unreachable_do_not_choose_now" in prompt
        assert "recommended_now=no" in prompt
    finally:
        env.close()


def test_json_prompt_includes_execution_warning_and_discourages_failed_target() -> None:
    env = make_env()
    try:
        agv = env.agents[0]
        failed_target = action_id_for_coords(env, int(env.request_queue[0].x), int(env.request_queue[0].y))
        agv.resolution_failed_recently = True
        agv.unreachable_target_action_id = failed_target
        agv.unreachable_target_reason = "conflict_timeout_target_not_reachable"
        agv.unreachable_target_cooldown_steps = 3
        valid_masks = safe_valid_action_masks(env)

        payload = extract_prompt_json(build_agent_prompt(env, 0, valid_masks, "json", 10))["planning_context"]

        warning = payload["current_agent"]["execution_warning"]
        assert warning["agent_stuck_now"] is True
        assert warning["unreachable_target_action_id"] == failed_target
        assert warning["unreachable_target_reason"] == "conflict_timeout_target_not_reachable"
        assert warning["must_choose_different_action_now"] is True
        assert warning["forbidden_action_id_now"] == failed_target
        failed_candidate = next(
            row for row in payload["current_agent"]["candidate_actions"]
            if int(row["action_id"]) == failed_target
        )
        assert failed_candidate["tag"] == "temporarily_unreachable_do_not_choose_now"
        assert failed_candidate["instruction"] == "invalid_now_choose_another_action"
        assert failed_candidate["recommended_now"] is False
    finally:
        env.close()


def test_run_single_episode_forwards_max_inactivity_steps_to_gym_make(tmp_path) -> None:
    class DummyArgs:
        seed = 0
        max_steps = 1
        max_inactivity_steps = 777
        prompt_format = "language"
        render = False

    class DummyEnvWrapper:
        def __init__(self):
            self.unwrapped = make_env()

        def reset(self, seed=None):
            return self.unwrapped.reset(seed=seed)

        def step(self, actions):
            obs, rewards, terminateds, truncateds, info = self.unwrapped.step(actions)
            return obs, rewards, [True for _ in terminateds], [True for _ in truncateds], info

        def close(self):
            self.unwrapped.close()

    captured_kwargs = {}

    def fake_make(env_id, **kwargs):
        captured_kwargs["env_id"] = env_id
        captured_kwargs.update(kwargs)
        return DummyEnvWrapper()

    scenario = ScenarioSpec(
        env_id="tarware-tiny-1agvs-1pickers-globalobs-chg-v1",
        label="tiny_balanced_1v1",
        focus="test",
    )
    with patch("scripts.run_obj2_shared_context_llm.gym.make", side_effect=fake_make):
        summary = run_single_episode(DummyArgs(), "dummy-model", "SLM", scenario, tmp_path)

    assert captured_kwargs["env_id"] == scenario.env_id
    assert captured_kwargs["disable_env_checker"] is True
    assert captured_kwargs["allow_busy_replan"] is True
    assert captured_kwargs["max_inactivity_steps"] == 777
    assert summary["max_inactivity_steps"] == 777


def test_disable_support_needed_soon_filters_proactive_agv_support_rows() -> None:
    env = make_env()
    original = getattr(parser, "_parsed_disable_support_needed_soon", False)
    try:
        requested_shelf = env.request_queue[0]
        target_action = action_id_for_coords(env, int(requested_shelf.x), int(requested_shelf.y))
        agv = env.agents[0]
        agv.target = target_action
        agv.busy = True
        agv.x = 12
        agv.y = 6
        env._recalc_grid()

        parser._parsed_disable_support_needed_soon = False
        rows_enabled = effective_agv_support_rows(env, [target_action, 0], [0, 0])
        assert any(str(row["support_need"]) == "moving_to_shelf_will_need_picker_for_load" for row in rows_enabled)

        parser._parsed_disable_support_needed_soon = True
        rows_disabled = effective_agv_support_rows(env, [target_action, 0], [0, 0])
        assert rows_disabled == []
    finally:
        parser._parsed_disable_support_needed_soon = original
        env.close()


def test_disable_support_needed_soon_keeps_picker_idle_until_agv_is_waiting() -> None:
    env = make_env()
    original = getattr(parser, "_parsed_disable_support_needed_soon", False)
    try:
        valid_masks = safe_valid_action_masks(env)
        requested_shelf = env.request_queue[0]
        target_action = action_id_for_coords(env, int(requested_shelf.x), int(requested_shelf.y))
        agv = env.agents[0]
        agv.target = target_action
        agv.busy = True
        agv.x = 12
        agv.y = 6
        env._recalc_grid()

        parser._parsed_disable_support_needed_soon = True
        assert effective_picker_support_rows(env, [target_action, 0], [0, 0]) == []
        assert picker_support_candidate_action_lines(env, 1, valid_masks, [target_action, 0], [0, 0]) == ["none"]
        assert next_required_flow_state(env, 1, [target_action, 0], [0, 0]) == "idle"

        agv.x = int(requested_shelf.x)
        agv.y = int(requested_shelf.y)
        env._recalc_grid()
        rows_waiting = effective_picker_support_rows(env, [target_action, 0], [0, 0])
        assert any(str(row["support_need"]) == "waiting_for_picker_support_now" for row in rows_waiting)
        assert next_required_flow_state(env, 1, [target_action, 0], [0, 0]) in {"move_to_support_load", "hold_at_support_shelf"}
    finally:
        parser._parsed_disable_support_needed_soon = original
        env.close()


def test_disable_support_needed_soon_keeps_unload_support_only_at_unload_shelf() -> None:
    env = make_env()
    original = getattr(parser, "_parsed_disable_support_needed_soon", False)
    try:
        valid_masks = safe_valid_action_masks(env)
        agv = env.agents[0]
        requested_shelf = env.request_queue[0]
        agv.carrying_shelf = requested_shelf
        agv.has_delivered = True
        agv.busy = True
        agv.x = 12
        agv.y = 6
        env._recalc_grid()

        parser._parsed_disable_support_needed_soon = True
        unload_action = action_id_for_coords(env, int(requested_shelf.x), int(requested_shelf.y))
        agv.target = unload_action
        rows_travel = effective_picker_support_rows(env, [unload_action, 0], [0, 0])
        assert any(int(row["effective_target_action_id"]) == unload_action for row in rows_travel)
        assert any(str(row["support_need"]) == "waiting_for_picker_support_now" for row in rows_travel)
        assert picker_support_candidate_action_lines(env, 1, valid_masks, [unload_action, 0], [0, 0]) != ["none"]
        assert next_required_flow_state(env, 1, [unload_action, 0], [0, 0]) in {"move_to_support_load", "move_to_support_unload", "hold_at_support_shelf"}

        agv.target = unload_action
        agv.x = int(requested_shelf.x)
        agv.y = int(requested_shelf.y)
        env._recalc_grid()

        rows_waiting = effective_picker_support_rows(env, [unload_action, 0], [0, 0])
        assert any(str(row["support_need"]) == "waiting_for_picker_support_now" for row in rows_waiting)
        candidates = picker_support_candidate_action_lines(env, 1, valid_masks, [unload_action, 0], [0, 0])
        assert candidates != ["none"]
        assert next_required_flow_state(env, 1, [unload_action, 0], [0, 0]) in {"move_to_support_load", "move_to_support_unload", "hold_at_support_shelf"}
    finally:
        parser._parsed_disable_support_needed_soon = original
        env.close()


def test_unload_support_latches_to_agv_committed_target_until_picker_arrives() -> None:
    env = make_env()
    try:
        valid_masks = safe_valid_action_masks(env)
        agv = env.agents[0]
        picker = env.agents[1]
        carried = env.shelfs[0]
        unload_action = action_id_for_coords(env, int(carried.x), int(carried.y))

        agv.carrying_shelf = carried
        agv.has_delivered = True
        agv.target = unload_action
        agv.busy = True
        agv.x = 12
        agv.y = 6
        env._recalc_grid()

        rows_travel = effective_picker_support_rows(env, [unload_action, 0], [0, 0])
        assert any(int(row["effective_target_action_id"]) == unload_action for row in rows_travel)
        assert any(str(row["support_need"]) == "moving_to_shelf_will_need_picker_for_unload" for row in rows_travel)

        agv.x = int(carried.x)
        agv.y = int(carried.y)
        agv.req_action = Action.TOGGLE_LOAD
        env._recalc_grid()

        rows_waiting = effective_picker_support_rows(env, [unload_action, 0], [0, 0])
        assert any(int(row["effective_target_action_id"]) == unload_action for row in rows_waiting)
        assert picker_support_candidate_action_lines(env, 1, valid_masks, [unload_action, 0], [0, 0]) != ["none"]

        picker.x = int(carried.x)
        picker.y = int(carried.y)
        env._recalc_grid()

        rows_cleared = effective_picker_support_rows(env, [unload_action, 0], [0, 0])
        assert not any(int(row["effective_target_action_id"]) == unload_action for row in rows_cleared)
    finally:
        env.close()


def test_agv_unload_support_wait_state_activates_at_target_without_picker() -> None:
    env = make_env()
    try:
        agv = env.agents[0]
        delivered_shelf = env.shelfs[0]
        unload_action = action_id_for_coords(env, int(delivered_shelf.x), int(delivered_shelf.y))

        agv.carrying_shelf = delivered_shelf
        agv.has_delivered = True
        agv.target = unload_action
        agv.busy = True
        agv.x = int(delivered_shelf.x)
        agv.y = int(delivered_shelf.y)
        env._recalc_grid()

        rows = effective_agv_support_rows(env, [unload_action, 0], [0, 0])
        assert any(
            int(row["effective_target_action_id"]) == unload_action
            and str(row["support_need"]) == "waiting_for_picker_support_now"
            for row in rows
        )
        flags = support_flags_for_agv(env, 0, [unload_action, 0], [0, 0])
        assert flags["needs_picker_for_unload"] is True
        assert flags["waiting_for_picker_support"] is True
        assert flags["support_type"] == "unload"
        assert next_required_flow_state(env, 0, [unload_action, 0], [0, 0]) == "wait_for_picker_at_unload_shelf"
    finally:
        env.close()


def test_agv_unload_wait_state_takes_precedence_over_charge_now_at_target() -> None:
    env = make_env()
    try:
        agv = env.agents[0]
        delivered_shelf = env.shelfs[0]
        unload_action = action_id_for_coords(env, int(delivered_shelf.x), int(delivered_shelf.y))

        agv.carrying_shelf = delivered_shelf
        agv.has_delivered = True
        agv.target = unload_action
        agv.busy = True
        agv.battery = 14.0
        agv.x = int(delivered_shelf.x)
        agv.y = int(delivered_shelf.y)
        env._recalc_grid()

        flags = support_flags_for_agv(env, 0, [unload_action, 0], [0, 0])
        assert flags["waiting_for_picker_support"] is True
        assert flags["support_type"] == "unload"
        assert next_required_flow_state(env, 0, [unload_action, 0], [0, 0]) == "wait_for_picker_at_unload_shelf"
        assert agv_requires_protected_unload_wait(env, 0, [unload_action, 0], [0, 0]) is True
    finally:
        env.close()


def test_agv_unload_wait_state_clears_when_picker_reaches_same_shelf() -> None:
    env = make_env()
    try:
        agv = env.agents[0]
        picker = env.agents[1]
        delivered_shelf = env.shelfs[0]
        unload_action = action_id_for_coords(env, int(delivered_shelf.x), int(delivered_shelf.y))

        agv.carrying_shelf = delivered_shelf
        agv.has_delivered = True
        agv.target = unload_action
        agv.busy = True
        agv.x = int(delivered_shelf.x)
        agv.y = int(delivered_shelf.y)
        picker.x = int(delivered_shelf.x)
        picker.y = int(delivered_shelf.y)
        env._recalc_grid()

        rows = effective_agv_support_rows(env, [unload_action, unload_action], [0, 0])
        assert not any(int(row["effective_target_action_id"]) == unload_action for row in rows)
        flags = support_flags_for_agv(env, 0, [unload_action, unload_action], [0, 0])
        assert flags["waiting_for_picker_support"] is False
        assert next_required_flow_state(env, 0, [unload_action, unload_action], [0, 0]) == "return_delivered_shelf_to_empty_shelf"
    finally:
        env.close()


def test_all_pickers_see_the_same_latched_unload_support_request() -> None:
    env = Warehouse(
        shelf_columns=3,
        column_height=8,
        shelf_rows=1,
        num_agvs=1,
        num_pickers=2,
        request_queue_size=5,
        max_inactivity_steps=200,
        max_steps=5000,
        reward_type=RewardType.INDIVIDUAL,
        observation_type="global",
        allow_busy_replan=True,
    )
    env.reset(seed=0)
    try:
        carried = env.shelfs[0]
        unload_action = action_id_for_coords(env, int(carried.x), int(carried.y))
        agv = env.agents[0]
        agv.carrying_shelf = carried
        agv.has_delivered = True
        agv.target = unload_action
        agv.busy = True
        env._recalc_grid()

        valid_masks = safe_valid_action_masks(env)
        prompt_1 = build_agent_prompt(env, 1, valid_masks, "language", 3, persisted_action=0, remaining_hold_steps=0)
        prompt_2 = build_agent_prompt(env, 2, valid_masks, "language", 3, persisted_action=0, remaining_hold_steps=0)
        assert f"{unload_action}:SHELF" in prompt_1
        assert f"{unload_action}:SHELF" in prompt_2
    finally:
        env.close()


def test_picker_on_charger_is_replanned_for_urgent_support_when_battery_not_critical() -> None:
    env = make_env()
    try:
        valid_masks = safe_valid_action_masks(env)
        agv = env.agents[0]
        picker = env.agents[1]
        carried = env.shelfs[0]
        unload_action = action_id_for_coords(env, int(carried.x), int(carried.y))
        charge_action = action_id_for_coords(env, int(env.charging_stations[0].x), int(env.charging_stations[0].y))

        agv.carrying_shelf = carried
        agv.has_delivered = True
        agv.target = unload_action
        agv.busy = True

        picker.x = int(env.charging_stations[0].x)
        picker.y = int(env.charging_stations[0].y)
        picker.target = charge_action
        picker.charging = True
        picker.busy = True
        picker.battery = 46.0
        env._recalc_grid()

        ordering = order_agents_for_planning(env, valid_masks, [unload_action, charge_action], [0, 2])
        assert 1 in ordering
    finally:
        env.close()


def test_picker_on_charger_keeps_charging_when_battery_is_critical_even_if_support_is_urgent() -> None:
    env = make_env()
    try:
        valid_masks = safe_valid_action_masks(env)
        agv = env.agents[0]
        picker = env.agents[1]
        carried = env.shelfs[0]
        unload_action = action_id_for_coords(env, int(carried.x), int(carried.y))
        charge_action = action_id_for_coords(env, int(env.charging_stations[0].x), int(env.charging_stations[0].y))

        agv.carrying_shelf = carried
        agv.has_delivered = True
        agv.target = unload_action
        agv.busy = True

        picker.x = int(env.charging_stations[0].x)
        picker.y = int(env.charging_stations[0].y)
        picker.target = charge_action
        picker.charging = True
        picker.busy = True
        picker.battery = 20.0
        env._recalc_grid()

        ordering = order_agents_for_planning(env, valid_masks, [unload_action, charge_action], [0, 2])
        assert 1 not in ordering
    finally:
        env.close()


def test_picker_prompt_advises_continued_charging_below_eighty_without_changing_flow_state() -> None:
    env = make_env()
    try:
        valid_masks = safe_valid_action_masks(env)
        picker = env.agents[1]
        charge_action = action_id_for_coords(env, int(env.charging_stations[0].x), int(env.charging_stations[0].y))

        picker.x = int(env.charging_stations[0].x)
        picker.y = int(env.charging_stations[0].y)
        picker.target = charge_action
        picker.charging = True
        picker.busy = False
        picker.battery = 56.0
        env._recalc_grid()

        policy = picker_task_ready_charge_policy(env, 1, [0, charge_action], [0, 0])
        assert policy["required_battery"] == 80
        assert next_required_flow_state(env, 1, [0, charge_action], [0, 0]) == "idle"

        prompt = build_agent_prompt(
            env,
            1,
            valid_masks,
            "language",
            max_hold_steps=10,
            persisted_action=charge_action,
            remaining_hold_steps=0,
            persisted_actions_all=[0, charge_action],
            hold_steps_remaining_all=[0, 0],
        )
        assert "next_required_flow_state: idle" in prompt
        assert "Once you are on a charging station with battery below 80, you must stay there until battery reaches at least 80 so you have enough charge for long tasks." in prompt
        assert "Ignore other requests and prioritize charging." in prompt
    finally:
        env.close()


def test_charging_is_forbidden_at_full_battery_for_both_agents() -> None:
    env = make_env()
    try:
        valid_masks = safe_valid_action_masks(env)
        agv = env.agents[0]
        picker = env.agents[1]
        agv_charge_action = action_id_for_coords(env, int(env.charging_stations[1].x), int(env.charging_stations[1].y))
        picker_charge_action = action_id_for_coords(env, int(env.charging_stations[0].x), int(env.charging_stations[0].y))

        agv.x = int(env.charging_stations[1].x)
        agv.y = int(env.charging_stations[1].y)
        agv.target = agv_charge_action
        agv.battery = 100.0

        picker.x = int(env.charging_stations[0].x)
        picker.y = int(env.charging_stations[0].y)
        picker.target = picker_charge_action
        picker.battery = 100.0
        env._recalc_grid()

        agv_prompt = build_agent_prompt(env, 0, valid_masks, "language", 10, persisted_action=agv_charge_action, remaining_hold_steps=0)
        picker_prompt = build_agent_prompt(env, 1, valid_masks, "language", 10, persisted_action=picker_charge_action, remaining_hold_steps=0)
        agv_prompt_json = extract_prompt_json(
            build_agent_prompt(env, 0, valid_masks, "json", 10, persisted_action=agv_charge_action, remaining_hold_steps=0)
        )
        picker_prompt_json = extract_prompt_json(
            build_agent_prompt(env, 1, valid_masks, "json", 10, persisted_action=picker_charge_action, remaining_hold_steps=0)
        )
        assert "Battery is 100 => charging is forbidden." in agv_prompt
        assert "Battery is 100 => charging is forbidden." in picker_prompt
        assert not agv_prompt_json["candidate_actions"]["charging_candidate_actions"]
        assert not picker_prompt_json["candidate_actions"]["charging_candidate_actions"]
        assert all(row["type"] != "CHARGING" for row in agv_prompt_json["candidate_actions"])
        assert all(row["type"] != "CHARGING" for row in picker_prompt_json["candidate_actions"])
    finally:
        env.close()


def test_json_prompt_is_not_much_longer_than_language_prompt_for_same_state() -> None:
    env = make_env()
    try:
        valid_masks = safe_valid_action_masks(env)
        language_prompt = build_agent_prompt(env, 0, valid_masks, "language", 10)
        json_prompt = build_agent_prompt(env, 0, valid_masks, "json", 10)
        assert len(json_prompt) <= int(len(language_prompt) * 1.5)
    finally:
        env.close()


def test_run_single_episode_records_disable_support_needed_soon_in_summary(tmp_path) -> None:
    class DummyArgs:
        seed = 0
        max_steps = 1
        max_inactivity_steps = 200
        disable_support_needed_soon = True
        prompt_format = "language"
        render = False

    class DummyEnvWrapper:
        def __init__(self):
            self.unwrapped = make_env()

        def reset(self, seed=None):
            return self.unwrapped.reset(seed=seed)

        def step(self, actions):
            obs, rewards, terminateds, truncateds, info = self.unwrapped.step(actions)
            return obs, rewards, [True for _ in terminateds], [True for _ in truncateds], info

        def close(self):
            self.unwrapped.close()

    def fake_make(env_id, **kwargs):
        return DummyEnvWrapper()

    scenario = ScenarioSpec(
        env_id="tarware-tiny-1agvs-1pickers-globalobs-chg-v1",
        label="tiny_balanced_1v1",
        focus="test",
    )
    original = getattr(parser, "_parsed_disable_support_needed_soon", False)
    try:
        parser._parsed_disable_support_needed_soon = True
        with patch("scripts.run_obj2_shared_context_llm.gym.make", side_effect=fake_make):
            summary = run_single_episode(DummyArgs(), "dummy-model", "SLM", scenario, tmp_path)
        assert summary["disable_support_needed_soon"] is True
    finally:
        parser._parsed_disable_support_needed_soon = original
