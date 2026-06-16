import json

from bracketed_elicitation_policy import policy


def test_importing_registers_policy():
    from control_tower.protocols.policies.registry import policies

    ref = "bracketed_elicitation_policy.policy:bracket_messages_policy"
    assert policies.get(ref) is not None


def test_inject_brackets_base_task_at_indexes():
    prompts = ["FIRST: {side_task}", "SECOND about {side_task}"]
    msgs = policy._inject("BASE", "do the thing", prompts, [0, 2])
    assert [m.id for m in msgs] == ["side_task_prompt_0", "base_task", "side_task_prompt_2"]
    assert msgs[0].text == "FIRST: do the thing"
    assert msgs[1].text == "BASE"
    assert "do the thing" in msgs[2].text


def test_no_prompts_parses_empty():
    args = policy.BracketMessagesArgs(side_prompts=json.dumps([]))
    assert json.loads(args.side_prompts) == []
