"""Exercise the actual query method without starting Kit or importing USD."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


source = Path(__file__).resolve().parents[2] / "robolab/core/world/world_state.py"
tree = ast.parse(source.read_text())
world_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WorldState")
method = next(n for n in world_class.body
              if isinstance(n, ast.FunctionDef) and n.name == "get_objects_in_contact_with")
namespace = {}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)


class World:
    get_objects_in_contact_with = namespace["get_objects_in_contact_with"]

    def __init__(self, contacts):
        self.env = SimpleNamespace(cfg=SimpleNamespace(physics_backend="newton_mjwarp"))
        self.contacts = contacts
        self.calls = []

    def resolve_contact_bodies(self, body):
        return ["left", "right"] if body == "both" else [body]

    def in_contact(self, body, obj, threshold, *, env_id):
        self.calls.append((body, obj, threshold, env_id))
        return (body, obj) in self.contacts


@pytest.mark.parametrize("candidates,expected", [
    (["rubiks_cube", "bowl"], ["rubiks_cube"]),
    (["bowl", "rubiks_cube"], ["rubiks_cube"]),
    (["bowl"], []),
    ([], []),
])
def test_newton_uses_object_identity_not_batch_position(candidates, expected):
    world = World({("gripper", "rubiks_cube")})
    assert world.get_objects_in_contact_with("gripper", candidates, .2, 3) == expected
    assert all(call[2:] == (.2, 3) for call in world.calls)


def test_alias_union_preserves_candidate_order_and_default_env():
    world = World({("left", "rubiks_cube"), ("right", "bowl")})
    assert world.get_objects_in_contact_with("both", ["bowl", "rubiks_cube"]) == ["bowl", "rubiks_cube"]
    assert all(call[3] == 0 for call in world.calls)


def test_physx_still_uses_existing_batch_path(monkeypatch):
    world = World({("gripper", "rubiks_cube")})
    world.env.cfg.physics_backend = "physx"
    world.env.scene = object()
    # Stop at batch lookup: this regression protects backend scoping without
    # requiring Torch/Kit to execute the unchanged PhysX implementation.
    def batch(scene, body):
        raise LookupError("physx batch lookup")
    monkeypatch.setitem(namespace, "get_batch_contact_sensor", batch)
    with pytest.raises(LookupError, match="physx batch lookup"):
        world.get_objects_in_contact_with("gripper", ["rubiks_cube", "bowl"])
