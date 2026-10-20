"""Audit logging. Register ``audit_after_agent`` as each LLM agent's
``after_agent_callback`` in the ADK layer. It records run_id, agent name, the
prompt hash and the playbook version; it stores IDs and hashes, never contract
text, and never alters the agent's output.

NOTE: check the callback signature against the installed google-adk 2.x version
at Milestone 0; attribute names below follow the documented CallbackContext.
"""

from __future__ import annotations

import json
import logging

from app.prompts import PROMPT_HASHES

log = logging.getLogger("review.audit")


def audit_after_agent(callback_context):
    st = getattr(callback_context, "state", {}) or {}
    agent = getattr(callback_context, "agent_name", "unknown")
    log.info(
        json.dumps(
            {
                "run_id": st.get("run_id"),
                "agent": agent,
                "prompt_hash": PROMPT_HASHES.get(agent),
                "playbook": st.get("playbook_version"),
            }
        )
    )
    return None  # do not alter the agent's output
