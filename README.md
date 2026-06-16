# bracketed-elicitation-policy

A small [Control Tower](https://github.com/linuxarena/control-tower) rollout policy that inserts
**caller-provided** messages around a task. It ships no prompt text and no model list — the
messages, model id, and generate-config are passed as policy args at launch, so they live in the
run config rather than in this repo.

Its purpose is to be pip-installable on a stock hawk runner: Control Tower resolves an external
`module:attr` `--policy` ref by importing the module (which registers the policy), so adding this
package to an eval-set's `packages` is enough to use it on hawk.

## Install

```bash
uv pip install "bracketed-elicitation-policy @ git+https://github.com/<org>/bracketed-elicitation-policy.git@main"
```

It imports `control_tower` and `inspect_ai` at runtime and declares no hard deps, so install it
**alongside** control-tower (which brings both).

## Use

```
--policy bracketed_elicitation_policy.policy:bracket_messages_policy
```

Policy args (all optional; the prompts are required only when a side task is present):

| arg | meaning |
|---|---|
| `side_prompts` | JSON array of message templates, each with a `{side_task}` field. Inserted around the base task. |
| `side_prompt_indexes` | comma-separated insertion points (default `0,2`). |
| `model_name` | exact inspect model id (e.g. `provider/model-name`); empty → CT `model` alias. |
| `generate_config` | JSON object passed to inspect `GenerateConfig`. |
| `max_steps` | graceful step budget (do **not** use an eval-level message/time limit). |

With no side task the conversation is just the base task; the `side_*` args are ignored.

Because the list/dict args are JSON strings, build the command programmatically rather than by hand.
