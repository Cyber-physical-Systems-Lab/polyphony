from typing import Any, Dict, Optional

import numpy as np

from state_translation_helper import (
    candidate_ids_for_prompt,
    describe_action_id_for_agent,
    get_requested_shelves,
    render_all_agents,
    render_self_state,
)


DEFAULT_FIXED_PROMPT_TEMPLATE = (
    "You are controlling one warehouse agent.\n"
    "Goal: maximize deliveries while keeping batteries safe.\n"
    "Use only valid candidate action ids.\n"
    "Choose a suitable Steps value.\n"
    "Fewer LLM calls improve performance, so use a larger Steps value when the plan is stable and unlikely to need revision.\n"
    "But do not overcommit: use a smaller Steps value when the environment may change soon, coordination may be needed, battery risk is rising, or urgent actions may appear.\n"
    "Step: {step}\n"
    "Agent: agent_{agent_id} ({agent_type})\n"
    "Self state: {self_state}\n"
    "Requested shelves:\n{requested_shelves}\n"
    "All agents:\n{all_agents}\n"
    "Candidate actions:\n{candidate_actions}\n"
    "Respond in plain text only:\n"
    "Reasoning: <short>\n"
    "Action: <action_id>\n"
    "Steps: <1..{max_hold_steps}>\n"
)


class _TemplateSafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def render_template(template: str, fields: Dict[str, Any]) -> str:
    """Render a prompt template while leaving unknown placeholders untouched."""
    return template.format_map(_TemplateSafeDict(**fields))


def build_fixed_prompt_fields(
    env,
    agent_idx: int,
    valid_masks: np.ndarray,
    step_count: int,
    max_candidate_ids: int,
    max_action_hold_steps: int,
) -> Dict[str, Any]:
    """Build the minimal prompt field set used by the fixed prompt template."""
    agent = env.agents[agent_idx]
    candidate_ids = candidate_ids_for_prompt(env, agent_idx, valid_masks, max_candidate_ids)
    requested = get_requested_shelves(env)
    all_agents = render_all_agents(env)

    return {
        "step": step_count,
        "agent_id": agent_idx,
        "agent_type": agent.type.name,
        "self_state": render_self_state(env, agent),
        "requested_shelves": "\n".join(f"- {row}" for row in requested) if requested else "- none",
        "all_agents": "\n".join(f"- {row}" for row in all_agents) if all_agents else "- none",
        "candidate_actions": (
            "\n".join(f"- {describe_action_id_for_agent(env, agent, action_id)}" for action_id in candidate_ids)
            if candidate_ids
            else "- 0:NOOP (distance_steps=0)"
        ),
        "max_hold_steps": max(1, int(max_action_hold_steps)),
    }


def build_fixed_prompt(
    env,
    agent_idx: int,
    valid_masks: np.ndarray,
    step_count: int,
    max_candidate_ids: int = 0,
    max_action_hold_steps: int = 10,
    template: Optional[str] = None,
) -> str:
    """Build a compact fixed-template prompt for one agent."""
    fields = build_fixed_prompt_fields(
        env=env,
        agent_idx=agent_idx,
        valid_masks=valid_masks,
        step_count=step_count,
        max_candidate_ids=max_candidate_ids,
        max_action_hold_steps=max_action_hold_steps,
    )
    return render_template(template or DEFAULT_FIXED_PROMPT_TEMPLATE, fields)
