"""P79: object + finger-pad friction as a run parameter (robolab/core/physics/friction.py).

Everything above the Isaac Lab import boundary is pure Python and is tested here; the
one thing these tests cannot prove -- that PhysX ends up holding the requested
coefficients -- is what ``friction_applied.json`` and ``verify_patches.py`` (P79) are for.
"""
import json
import pathlib
import re
import sys
import types

import pytest

import robolab.constants
from robolab.core.physics import friction as F

ROOT = pathlib.Path(__file__).resolve().parents[1]

CATALOG = [
    {"name": "banana", "usd_path": "assets/objects/ycb/banana.usd", "class": "fruit",
     "static_friction": 2.0, "dynamic_friction": 2.0, "restitution": 0.1},
    {"name": "bowl", "usd_path": "assets/objects/ycb/bowl.usd", "class": "container",
     "static_friction": 2.0, "dynamic_friction": 2.0, "restitution": 0.3},
    {"name": "bowl", "usd_path": "assets/objects/ycb/bowl2.usd", "class": "container",
     "static_friction": 2.0, "dynamic_friction": 2.0, "restitution": 0.3},
    {"name": "widget", "usd_path": "assets/objects/objaverse/widget.usd", "class": "",
     "static_friction": 10.0, "dynamic_friction": 10.0, "restitution": None},
]
SCENE = [
    {"name": "table", "payload": ["../fixtures/table_oak.usd"], "rigid_body": True},
    {"name": "banana", "payload": ["../objects/ycb/banana.usd"], "rigid_body": True},
    {"name": "bowl_1", "payload": ["../objects/ycb/bowl2.usd"], "rigid_body": True},
    {"name": "widget", "payload": ["../objects/objaverse/widget.usd"], "rigid_body": True},
    {"name": "mystery", "payload": [], "rigid_body": True},
]
OBJECTS = ["banana", "bowl_1", "widget", "mystery"]


# --------------------------------------------------------------------------- parsing

@pytest.mark.parametrize("text", [None, "", "upstream", "UPSTREAM", "  upstream "])
def test_upstream_is_the_default_and_means_untouched(text):
    spec = F.parse_friction(text)
    assert spec.mode == "upstream" and not spec.active
    assert F.resolve_materials(spec, SCENE, OBJECTS, CATALOG) == {}
    assert F.pad_material(spec) is None


def test_default_constant_is_upstream():
    assert robolab.constants.FRICTION == "upstream"


def test_a_number_is_a_uniform_coefficient():
    spec = F.parse_friction("0.5")
    assert spec.mode == "uniform" and spec.uniform == 0.5 and spec.active
    mats = F.resolve_materials(spec, SCENE, OBJECTS, CATALOG)
    assert set(mats) == set(OBJECTS)
    assert all(m.static == 0.5 and m.dynamic == 0.5 and m.source == "uniform" for m in mats.values())
    # restitution is carried over from the catalog, so *only* friction changes
    assert mats["banana"].restitution == 0.1 and mats["bowl_1"].restitution == 0.3
    assert mats["widget"].restitution == 0.0 and mats["mystery"].restitution == 0.0
    assert F.pad_material(spec) == F.Material(0.5, 0.5, 0.0, "uniform")


@pytest.mark.parametrize("bad", ["-1", "25", "nope", "missing.json"])
def test_bad_values_are_rejected_before_isaac_boots(bad):
    with pytest.raises(ValueError):
        F.parse_friction(bad)


def test_the_realistic_preset_loads_and_is_internally_consistent():
    spec = F.parse_friction("realistic")
    assert spec.mode == "table" and spec.name == "realistic"
    for key, entry in spec.table.items():
        assert 0.0 < entry["dynamic"] <= entry["static"] <= 1.0, key
    g = spec.table["gripper"]
    # effective pad-object coefficient (PhysX average rule) for every class stays in a
    # dry-rubber-on-solids range, i.e. the preset is not secretly an upstream lookalike
    for key, entry in spec.table.items():
        if key == "gripper":
            continue
        assert 0.3 <= (g["static"] + entry["static"]) / 2 <= 1.0, key


def test_the_realistic_preset_covers_every_class_in_the_catalog():
    catalog = json.load(open(ROOT / "assets/objects/object_catalog.json"))
    table = F.parse_friction("realistic").table
    classes = {row.get("class") or "" for row in catalog}
    uncovered = sorted(c for c in classes if c and c not in table)
    assert not uncovered, f"catalog classes falling through to 'default': {uncovered}"


def test_a_user_table_needs_gripper_and_default_and_dynamic_le_static(tmp_path):
    ok = tmp_path / "t.json"
    ok.write_text(json.dumps({"_comment": "x", "gripper": 0.9, "default": {"static": 0.5, "dynamic": 0.4}, "fruit": 0.3}))
    spec = F.parse_friction(str(ok))
    assert spec.table["gripper"] == {"static": 0.9, "dynamic": 0.9}
    assert spec.table["default"] == {"static": 0.5, "dynamic": 0.4}
    for broken in ({"default": 0.5}, {"gripper": 0.5}, {"gripper": 0.5, "default": {"static": 0.3, "dynamic": 0.4}},
                   {"gripper": True, "default": 0.5}):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(broken))
        with pytest.raises(ValueError):
            F.parse_friction(str(bad))


# --------------------------------------------------------------------------- resolution

def test_payload_key_and_catalog_index_match_on_the_usd_path_not_the_name():
    assert F.payload_key(["../objects/ycb/bowl2.usd"]) == "ycb/bowl2.usd"
    assert F.payload_key("assets/objects/ycb/bowl.usd") == "ycb/bowl.usd"
    assert F.payload_key([]) is None and F.payload_key(None) is None
    idx = F.catalog_index(CATALOG)
    assert idx["ycb/bowl2.usd"]["restitution"] == 0.3 and len(idx) == 4  # two 'bowl' rows, both kept


def test_table_resolution_by_class_with_default_fallback(tmp_path):
    t = tmp_path / "t.json"
    t.write_text(json.dumps({"gripper": 0.8, "default": {"static": 0.4, "dynamic": 0.35},
                             "fruit": {"static": 0.3, "dynamic": 0.2}, "container": 0.6}))
    mats = F.resolve_materials(F.parse_friction(str(t)), SCENE, OBJECTS, CATALOG)
    assert mats["banana"] == F.Material(0.3, 0.2, 0.1, "class:fruit")
    assert mats["bowl_1"] == F.Material(0.6, 0.6, 0.3, "class:container")
    assert mats["widget"].source == "default" and mats["widget"].static == 0.4       # empty class
    assert mats["mystery"].source == "default" and mats["mystery"].restitution == 0.0  # not in the catalog
    assert F.pad_material(F.parse_friction(str(t))) == F.Material(0.8, 0.8, 0.0, "gripper")


def test_the_tabletop_is_a_rigid_object_and_resolves_as_a_fixture():
    """PhysX readback on the pod (2026-08-28): `table` is in env.scene.rigid_objects at the
    PhysX default 0.5/0.5 -- it has no catalog row, so it must not fall through to `default`."""
    mats = F.resolve_materials(F.parse_friction("realistic"), SCENE, ["table"], CATALOG)
    assert mats["table"].source == "class:fixture" and mats["table"].static == 0.5


def test_only_isaac_assets_get_a_term_even_if_the_scene_lists_more():
    mats = F.resolve_materials(F.parse_friction("1.0"), SCENE, ["banana"], CATALOG)
    assert list(mats) == ["banana"]   # 'table' is in SCENE but is not a RigidObjectCfg


# --------------------------------------------------------------------------- event terms

def test_event_params_set_rather_than_randomise():
    p = F.event_params(F.Material(0.4, 0.35, 0.1, "x"))
    assert p["static_friction_range"] == (0.4, 0.4) and p["dynamic_friction_range"] == (0.35, 0.35)
    assert p["restitution_range"] == (0.1, 0.1) and p["num_buckets"] == 1 and p["make_consistent"] is True


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __eq__(self, other):
        return type(self) is type(other) and self.__dict__ == other.__dict__


def _stub_isaaclab(monkeypatch):
    """The three Isaac Lab names the module imports lazily, as recording stand-ins."""
    def rrbm(*a, **k):
        raise AssertionError("never called at cfg build time")
    mdp = types.ModuleType("isaaclab.envs.mdp"); mdp.randomize_rigid_body_material = rrbm
    envs = types.ModuleType("isaaclab.envs"); envs.mdp = mdp
    managers = types.ModuleType("isaaclab.managers")

    class EventTermCfg(_Cfg):
        pass

    class SceneEntityCfg(_Cfg):
        def __init__(self, name, body_names=None):
            super().__init__(name=name, body_names=body_names)
    managers.EventTermCfg = EventTermCfg; managers.SceneEntityCfg = SceneEntityCfg
    assets = types.ModuleType("isaaclab.assets")

    class RigidObjectCfg(_Cfg):
        pass
    assets.RigidObjectCfg = RigidObjectCfg
    root = types.ModuleType("isaaclab"); root.envs = envs; root.managers = managers; root.assets = assets
    for name, mod in (("isaaclab", root), ("isaaclab.envs", envs), ("isaaclab.envs.mdp", mdp),
                      ("isaaclab.managers", managers), ("isaaclab.assets", assets)):
        monkeypatch.setitem(sys.modules, name, mod)
    return rrbm, EventTermCfg, SceneEntityCfg, RigidObjectCfg


def test_event_terms_one_per_object_plus_the_pads(monkeypatch):
    rrbm, EventTermCfg, SceneEntityCfg, _ = _stub_isaaclab(monkeypatch)
    mats = {"banana": F.Material(0.5, 0.5, 0.1, "uniform"), "bowl_1": F.Material(0.5, 0.5, 0.3, "uniform")}
    terms = F.event_terms(mats, F.Material(0.5, 0.5, 0.0, "uniform"), ["left_inner_finger", "right_inner_finger"])
    assert sorted(terms) == ["friction_banana", "friction_bowl_1", "friction_robot_pads"]
    t = terms["friction_banana"]
    assert t.func is rrbm and t.mode == "startup"
    assert t.params["asset_cfg"] == SceneEntityCfg("banana") and t.params["restitution_range"] == (0.1, 0.1)
    pads = terms["friction_robot_pads"].params["asset_cfg"]
    assert pads.name == "robot" and pads.body_names == ["left_inner_finger", "right_inner_finger"]
    # no pad bodies declared -> objects only, no pad term
    assert "friction_robot_pads" not in F.event_terms(mats, F.Material(0.5, 0.5, 0.0, "uniform"), None)


def _fake_env_cfg(RigidObjectCfg):
    scene = _Cfg(scene=_Cfg(spawn=_Cfg(usd_path="/scenes/banana_bowl.usda")),
                 banana=RigidObjectCfg(), bowl_1=RigidObjectCfg(), table=_Cfg(), robot=_Cfg())
    return _Cfg(scene=scene, events=_Cfg(reset="keep-me"))


def test_install_upstream_stamps_provenance_and_adds_nothing(monkeypatch):
    _, _, _, RigidObjectCfg = _stub_isaaclab(monkeypatch)
    usd = types.ModuleType("robolab.core.utils.usd_utils")
    usd.get_usd_objects_info = lambda path: (_ for _ in ()).throw(AssertionError("scene must not be parsed under upstream"))
    monkeypatch.setitem(sys.modules, "robolab.core.utils.usd_utils", usd)
    cfg = _fake_env_cfg(RigidObjectCfg)
    prov = F.install(cfg, "/scenes/banana_bowl.usda", ["left_inner_finger"], spec=F.parse_friction("upstream"), catalog=CATALOG)
    assert prov["mode"] == "upstream" and prov["objects"] == {} and prov["gripper"] is None
    assert cfg.friction is prov and vars(cfg.events) == {"reset": "keep-me"}


def test_install_uniform_adds_startup_terms_and_full_provenance(monkeypatch):
    _, _, _, RigidObjectCfg = _stub_isaaclab(monkeypatch)
    usd = types.ModuleType("robolab.core.utils.usd_utils"); usd.get_usd_objects_info = lambda path: SCENE
    monkeypatch.setitem(sys.modules, "robolab.core.utils.usd_utils", usd)
    cfg = _fake_env_cfg(RigidObjectCfg)
    prov = F.install(cfg, "/scenes/banana_bowl.usda", ["left_inner_finger", "right_inner_finger"],
                     spec=F.parse_friction("0.5"), catalog=CATALOG)
    assert sorted(vars(cfg.events)) == ["friction_banana", "friction_bowl_1", "friction_robot_pads", "reset"]
    assert prov["mode"] == "uniform" and prov["spec"] == "0.5"
    assert prov["objects"]["banana"] == {"static": 0.5, "dynamic": 0.5, "restitution": 0.1, "source": "uniform"}
    assert prov["gripper"]["static"] == 0.5 and prov["gripper_bodies"] == ["left_inner_finger", "right_inner_finger"]
    assert json.dumps(prov)  # serialisable into env_cfg.json


def test_install_reads_the_constant_when_no_spec_is_given(monkeypatch):
    _, _, _, RigidObjectCfg = _stub_isaaclab(monkeypatch)
    usd = types.ModuleType("robolab.core.utils.usd_utils"); usd.get_usd_objects_info = lambda path: SCENE
    monkeypatch.setitem(sys.modules, "robolab.core.utils.usd_utils", usd)
    monkeypatch.setattr(robolab.constants, "FRICTION", "realistic")
    cfg = _fake_env_cfg(RigidObjectCfg)
    prov = F.install(cfg, "/scenes/x.usda", ["left_inner_finger"], catalog=CATALOG)
    assert prov["mode"] == "table" and prov["objects"]["banana"]["source"] == "class:fruit"


# --------------------------------------------------------------------------- readback check

def _applied(objects, pads, requested):
    return {"objects": objects, "gripper": pads, "requested": requested}


REQ = {"mode": "uniform", "spec": "0.5", "gripper_bodies": ["left_inner_finger", "right_inner_finger"],
       "gripper": {"static": 0.5, "dynamic": 0.5, "restitution": 0.0, "source": "uniform"},
       "objects": {"banana": {"static": 0.5, "dynamic": 0.5, "restitution": 0.1, "source": "uniform"}}}


def test_check_applied_passes_only_when_physx_holds_the_request():
    good = _applied({"banana": [[0.5, 0.5, 0.1], [0.5, 0.5, 0.1]]},
                    {"left_inner_finger": [[0.5, 0.5, 0.0]], "right_inner_finger": [[0.5, 0.5, 0.0]]}, REQ)
    assert F.check_applied(good) == []
    stale = _applied({"banana": [[2.0, 2.0, 0.1]]}, {"left_inner_finger": [[0.5, 0.5, 0.0]], "right_inner_finger": [[0.5, 0.5, 0.0]]}, REQ)
    assert F.check_applied(stale) == ["banana: requested 0.5/0.5, PhysX holds 2.0/2.0"]
    pad_missed = _applied({"banana": [[0.5, 0.5, 0.1]]}, {"left_inner_finger": [[0.5, 0.5, 0.0]], "right_inner_finger": "NOT A BODY"}, REQ)
    assert F.check_applied(pad_missed) == ["pad right_inner_finger: no PhysX readback (NOT A BODY)"]
    one_shape_off = _applied({"banana": [[0.5, 0.5, 0.1], [2.0, 2.0, 0.1]]},
                             {"left_inner_finger": [[0.5, 0.5, 0.0]], "right_inner_finger": [[0.5, 0.5, 0.0]]}, REQ)
    assert len(F.check_applied(one_shape_off)) == 1


def test_verify_patches_p79_judges_from_the_readback(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import verify_patches as vp
    run = tmp_path / "rc7_X"; (run / "XTask").mkdir(parents=True)
    assert vp.p79_friction_applied(str(run)).verdict == "N/A"           # predates P79
    good = _applied({"banana": [[0.5, 0.5, 0.1]]},
                    {"left_inner_finger": [[0.5, 0.5, 0.0]], "right_inner_finger": [[0.5, 0.5, 0.0]]}, REQ)
    (run / "XTask" / "friction_applied.json").write_text(json.dumps(good))
    r = vp.p79_friction_applied(str(run))
    assert r.verdict == "PASS" and r.opportunities == 3, r
    bad = dict(good); bad["objects"] = {"banana": [[2.0, 2.0, 0.1]]}
    (run / "XTask" / "friction_applied.json").write_text(json.dumps(bad))
    assert vp.p79_friction_applied(str(run)).verdict == "FAIL"
    upstream = _applied({"banana": [[2.0, 2.0, 0.1]]}, {"left_inner_finger": [[2.0, 2.0, 0.0]]},
                        {"mode": "upstream", "spec": "upstream", "objects": {}, "gripper": None, "gripper_bodies": ["left_inner_finger"]})
    (run / "XTask" / "friction_applied.json").write_text(json.dumps(upstream))
    r = vp.p79_friction_applied(str(run))
    assert r.verdict == "N/A" and "2/2 x1" in r.detail   # baseline is reported as a measurement, never as a pass


# --------------------------------------------------------------------------- wiring (static)

def test_every_runner_gets_the_flag_through_add_common_eval_args():
    runner = (ROOT / "robolab/eval/runner.py").read_text()
    assert '"--friction"' in runner and "robolab.constants.FRICTION = " in runner
    assert "type=_friction_arg" in runner   # validated at parse time, before Isaac boots


def test_every_robot_that_declares_ee_recorder_bodies_also_declares_friction_bodies():
    missing = []
    for path in sorted((ROOT / "robolab/robots").glob("*.py")):
        text = path.read_text()
        for cls in re.findall(r"^(\w+Cfg)\.ee_recorder_bodies\s*=", text, re.M):
            if not re.search(rf"^{cls}\.friction_bodies\s*=", text, re.M):
                missing.append(f"{path.name}:{cls}")
    assert not missing, f"robots without a friction_bodies label (pads would silently keep 2.0): {missing}"


def test_droid_pads_are_the_prims_that_carry_the_robotiq_material():
    # The two prims bound to /panda/Gripper/Robotiq_2F_85/PhysicsMaterial in the robot USD
    # (read with pxr on 2026-08-28) -- the override must land on exactly these bodies.
    text = (ROOT / "robolab/robots/droid.py").read_text()
    assert 'DroidCfg.friction_bodies = ["left_inner_finger", "right_inner_finger"]' in text


def test_every_robot_label_is_shadowed_in_the_generated_scene_cfg():
    """Robot cfg labels (`XCfg.<label> = ...`) are class attributes on a base of the generated
    scene cfg; `@configclass` copies them onto the instance and `InteractiveScene` then raises
    `Unknown asset config type for <label>` unless generate_scene_env_cfg shadows the label
    with None. P79's first GPU probe died exactly this way (2026-08-28), 2 min into the boot."""
    config = (ROOT / "robolab/core/environments/config.py").read_text()
    members = config.split("members = {", 1)[1].split("}", 1)[0]
    shadowed = set(re.findall(r'"(\w+)":\s*(?:None|table_fixture_asset)', members))
    labels = set()
    for path in (ROOT / "robolab/robots").glob("*.py"):
        labels |= set(re.findall(r"^\w+Cfg\.(\w+)\s*=", path.read_text(), re.M))
    assert labels - shadowed == set(), f"robot labels not shadowed to None in generate_scene_env_cfg: {sorted(labels - shadowed)}"
