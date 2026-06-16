"""Pip-installable home for Control Tower rollout building blocks that insert caller-provided
messages around the task. Importing the package fires the registrations below (Control Tower
resolves an external ``module:attr`` ref by importing the module).

- ``bracket_messages_policy`` (``--policy``) in the ``policies`` registry.
- ``bracketed-agent`` blue protocol (``--protocol``) that seeds the bracketed conversation, plus the
  ``passthrough`` untrusted policy (``--untrusted-policy``) that carries an exact model id +
  generate-config and edits nothing.

Prompts, model id, and generate-config are supplied as args at launch time, not bundled here.
"""

from bracketed_elicitation_policy.policy import bracket_messages_policy
from bracketed_elicitation_policy.protocol import (
    bracketed_agent_protocol,
    passthrough_policy,
)

__all__ = [
    "bracket_messages_policy",
    "bracketed_agent_protocol",
    "passthrough_policy",
]
