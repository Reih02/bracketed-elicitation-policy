import json

from bracketed_elicitation_policy import protocol


def test_eval2_components_register():
    from control_tower.protocols.blue_protocols.registry import blue_protocols
    from control_tower.protocols.untrusted_policy.registry import untrusted_policies

    assert blue_protocols.get("bracketed-agent") is not None
    assert untrusted_policies.get("passthrough") is not None


def _metadata(side):
    from control_tower.environments.environment import EnvironmentData
    from control_tower.task.sample_metadata import (
        MainTaskData,
        SampleMetadata,
        SideTaskData,
    )

    return SampleMetadata(
        main_task=MainTaskData(id="m", name="m", description="BASE", environment="env"),
        side_task=side,
        environment=EnvironmentData(id="env", display_name="Env", description="ENV DESC"),
    )


class _FakeState:
    def __init__(self, md):
        self._md = md

    def metadata_as(self, cls):
        return self._md


def test_seed_brackets_with_ids_and_system_first():
    from control_tower.task.sample_metadata import SideTaskData

    side = SideTaskData(id="s", name="s", environment="env", goal="GOAL", scoring_details="DET")
    prompts = json.dumps(["A {side_task}", "B {side_task}"])
    msgs = protocol._seed_messages(_FakeState(_metadata(side)), prompts, "0,2")
    assert msgs[0].role == "system" and "ENV DESC" in msgs[0].text
    assert [m.id for m in msgs[1:]] == ["side_task_prompt_0", "base_task", "side_task_prompt_2"]
    assert "GOAL" in msgs[1].text and "DET" in msgs[1].text


def test_seed_honest_is_just_system_and_base():
    msgs = protocol._seed_messages(_FakeState(_metadata(None)), "[]", "0,2")
    assert [m.role for m in msgs] == ["system", "user"]
    assert msgs[1].id == "base_task"
