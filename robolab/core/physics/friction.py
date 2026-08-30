"""Object and gripper friction as a *run parameter* (RoboLab Verified, P79).

What upstream ships
-------------------
Friction is baked into the assets. Every object USD carries its own
``PhysicsMaterial`` and, read back from the files (``assets/objects/object_catalog.json``),
289 of 312 objects have static = dynamic = 2.0, nine fruit have 5.0, seven Objaverse
objects have 10.0, and the Robotiq finger pads
(``assets/robots/franka_robotiq_2f_85_flattened.usd``) have 2.0. PhysX combines two
materials with the *average* rule, so the pad-object coefficient is 2.0 for most
objects, 3.5 for the fruit and 6.0 for a bagel. Dry rubber on plastic is 0.5-1.0.

What this module does
---------------------
Nothing, unless asked. ``robolab.constants.FRICTION`` (``--friction`` on every eval
runner) selects one of

* ``upstream``  -- the authored materials, untouched. **The benchmark default.**
* ``<number>``  -- one coefficient for every object *and* the finger pads
  (static = dynamic = number; the effective pad-object coefficient is the same number).
* ``realistic`` -- the bundled per-class table ``friction_realistic.json``.
* ``<path>.json`` -- a user table in the same format.

The override is applied at environment start-up through Isaac Lab's own
``randomize_rigid_body_material`` event term with a degenerate range ``(mu, mu)`` and one
bucket, i.e. it *sets* the PhysX shape materials; the USD files are never edited and a
run with ``upstream`` is bit-identical to a run without this module. Restitution is
carried over from the catalog so only friction changes.

Provenance, so a number can be traced back
------------------------------------------
* ``env_cfg.friction`` (serialised into ``env_cfg.json``) records what was *requested*
  per object, with the catalog class it was resolved from.
* ``friction_applied.json`` next to it records what PhysX *reports* after start-up
  (:func:`read_applied`), for every object shape and for the finger pads -- including
  under ``upstream``, so the baseline's 2.0 is a measurement, not a claim.
* ``scripts/verify_patches.py`` (P79) compares the two.

Everything above the Isaac Lab import boundary is pure Python and covered by
``offline_tests/test_p79_friction.py``.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import robolab.constants

UPSTREAM = "upstream"
GRIPPER_KEY = "gripper"
DEFAULT_KEY = "default"
EVENT_PREFIX = "friction_"
APPLIED_FILENAME = "friction_applied.json"
PRESETS = {
    "realistic": os.path.join(os.path.dirname(os.path.abspath(__file__)), "friction_realistic.json"),
}


@dataclass(frozen=True)
class Material:
    static: float
    dynamic: float
    restitution: float
    source: str  # "uniform" | "class:<name>" | "default" | "gripper"


@dataclass(frozen=True)
class FrictionSpec:
    mode: str                     # "upstream" | "uniform" | "table"
    text: str                     # what the user typed, for provenance
    uniform: float | None = None
    table: dict | None = None     # class -> {"static", "dynamic"}
    name: str | None = None       # preset name or table path

    @property
    def active(self) -> bool:
        return self.mode != UPSTREAM


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def _coeff_pair(value, where: str) -> tuple[float, float]:
    """A table entry is a number (static = dynamic) or {"static": s, "dynamic": d}."""
    if isinstance(value, bool):
        raise ValueError(f"friction table entry {where!r} is a bool")
    if isinstance(value, (int, float)):
        s = d = float(value)
    elif isinstance(value, dict) and "static" in value:
        s = float(value["static"])
        d = float(value.get("dynamic", s))
    else:
        raise ValueError(f"friction table entry {where!r} must be a number or {{static, dynamic}}, got {value!r}")
    for name, v in (("static", s), ("dynamic", d)):
        if not (0.0 <= v <= 20.0):
            raise ValueError(f"friction table entry {where!r}: {name} = {v} is outside [0, 20]")
    if d > s:
        raise ValueError(f"friction table entry {where!r}: dynamic ({d}) exceeds static ({s})")
    return s, d


def load_table(path: str) -> dict[str, dict[str, float]]:
    with open(path) as f:
        raw = json.load(f)
    table = {}
    for key, value in raw.items():
        if key.startswith("_"):
            continue
        s, d = _coeff_pair(value, key)
        table[key] = {"static": s, "dynamic": d}
    for required in (GRIPPER_KEY, DEFAULT_KEY):
        if required not in table:
            raise ValueError(f"friction table {path} has no {required!r} entry")
    return table


def parse_friction(text: str | None) -> FrictionSpec:
    """``None``/``""``/``upstream`` -> untouched; a number -> uniform; a preset name or a
    ``.json`` path -> table."""
    raw = (text or "").strip()
    if raw == "" or raw.lower() == UPSTREAM:
        return FrictionSpec(mode=UPSTREAM, text=UPSTREAM)
    try:
        mu = float(raw)
    except ValueError:
        mu = None
    if mu is not None:
        if not (0.0 <= mu <= 20.0):
            raise ValueError(f"--friction {raw}: coefficient must be within [0, 20]")
        return FrictionSpec(mode="uniform", text=raw, uniform=mu)
    path = PRESETS.get(raw.lower(), raw)
    if not os.path.isfile(path):
        raise ValueError(
            f"--friction {raw!r}: not a number, not a preset ({', '.join(sorted(PRESETS))}) "
            f"and not a readable .json table")
    return FrictionSpec(mode="table", text=raw, table=load_table(path), name=raw)


# --------------------------------------------------------------------------- #
# Resolution: which coefficient does each scene object get?
# --------------------------------------------------------------------------- #

def payload_key(payload) -> str | None:
    """``'../objects/ycb/banana.usd'`` -> ``'ycb/banana.usd'`` (the part under ``objects/``)."""
    if isinstance(payload, (list, tuple)):
        payload = payload[0] if payload else None
    if not payload:
        return None
    p = str(payload).replace("\\", "/")
    marker = "objects/"
    i = p.rfind(marker)
    return p[i + len(marker):] if i >= 0 else os.path.basename(p)


def catalog_index(catalog: list[dict]) -> dict[str, dict]:
    """Catalog rows keyed the same way as :func:`payload_key`. Names are not unique in the
    catalog (``bowl``, ``mug``, ...); the USD path is."""
    return {payload_key(row.get("usd_path")): row for row in catalog if row.get("usd_path")}


def load_catalog(path: str | None = None) -> list[dict]:
    with open(path or robolab.constants.OBJECT_CATALOG_PATH) as f:
        return json.load(f)


def resolve_materials(spec: FrictionSpec, scene_objects: list[dict], object_names: list[str],
                      catalog: list[dict]) -> dict[str, Material]:
    """One :class:`Material` per rigid scene object that will receive an override.

    ``scene_objects`` is what ``get_usd_objects_info`` returns for the scene (name,
    payload, ...); ``object_names`` are the scene objects that exist as Isaac Lab
    assets (``RigidObjectCfg``) -- only those can be addressed by an event term.
    """
    if not spec.active:
        return {}
    index = catalog_index(catalog)
    by_name = {o.get("name"): o for o in scene_objects}
    out: dict[str, Material] = {}
    for name in object_names:
        info = by_name.get(name, {})
        row = index.get(payload_key(info.get("payload")))
        restitution = row.get("restitution") if row else None
        restitution = float(restitution) if restitution is not None else 0.0
        if spec.mode == "uniform":
            out[name] = Material(spec.uniform, spec.uniform, restitution, "uniform")
            continue
        cls = (row or {}).get("class") or ""
        if not cls and "fixtures/" in str(info.get("payload") or ""):
            cls = "fixture"       # the tabletop (`../fixtures/table_oak.usd`) is a rigid object with no catalog row
        entry = spec.table.get(cls) if cls else None
        if entry is None:
            entry, source = spec.table[DEFAULT_KEY], ("default" if not cls else f"default(class:{cls})")
        else:
            source = f"class:{cls}"
        out[name] = Material(entry["static"], entry["dynamic"], restitution, source)
    return out


def pad_material(spec: FrictionSpec) -> Material | None:
    """The finger pads' material under this spec (restitution stays 0, as authored)."""
    if not spec.active:
        return None
    if spec.mode == "uniform":
        return Material(spec.uniform, spec.uniform, 0.0, "uniform")
    g = spec.table[GRIPPER_KEY]
    return Material(g["static"], g["dynamic"], 0.0, GRIPPER_KEY)


# --------------------------------------------------------------------------- #
# Event terms
# --------------------------------------------------------------------------- #

def event_params(material: Material) -> dict:
    """The ``randomize_rigid_body_material`` params that *set* (not randomise) a material:
    a degenerate range and a single bucket."""
    return {
        "static_friction_range": (material.static, material.static),
        "dynamic_friction_range": (material.dynamic, material.dynamic),
        "restitution_range": (material.restitution, material.restitution),
        "num_buckets": 1,
        "make_consistent": True,
    }


def event_terms(materials: dict[str, Material], pad: Material | None,
                pad_bodies: list[str] | None, robot_name: str = "robot") -> dict:
    """Build the Isaac Lab ``EventTermCfg`` objects. Imports Isaac Lab lazily so the rest of
    this module stays importable without a simulator."""
    from isaaclab.envs import mdp
    from isaaclab.managers import EventTermCfg, SceneEntityCfg

    terms = {}
    for name, mat in materials.items():
        terms[f"{EVENT_PREFIX}{name}"] = EventTermCfg(
            func=mdp.randomize_rigid_body_material, mode="startup",
            params={**event_params(mat), "asset_cfg": SceneEntityCfg(name)})
    if pad is not None and pad_bodies:
        terms[f"{EVENT_PREFIX}{robot_name}_pads"] = EventTermCfg(
            func=mdp.randomize_rigid_body_material, mode="startup",
            params={**event_params(pad), "asset_cfg": SceneEntityCfg(robot_name, body_names=list(pad_bodies))})
    return terms


def provenance(spec: FrictionSpec, materials: dict[str, Material], pad: Material | None,
               pad_bodies: list[str] | None) -> dict:
    """What goes into ``env_cfg.friction`` (and therefore ``env_cfg.json``)."""
    return {
        "mode": spec.mode,
        "spec": spec.text,
        "table": spec.name,
        "objects": {name: asdict(m) for name, m in sorted(materials.items())},
        "gripper": asdict(pad) if pad else None,
        "gripper_bodies": list(pad_bodies) if pad_bodies else [],
    }


def install(env_cfg, scene_path: str, pad_bodies: list[str] | None,
            spec: FrictionSpec | None = None, catalog: list[dict] | None = None) -> dict:
    """Attach the friction override to a generated env cfg. Called from
    ``GeneratedTaskEnvCfg.__post_init__``; a no-op that still stamps provenance under
    ``upstream``."""
    from isaaclab.assets import RigidObjectCfg
    from robolab.core.utils.usd_utils import get_usd_objects_info

    spec = spec or parse_friction(robolab.constants.FRICTION)
    object_names = [name for name, value in vars(env_cfg.scene).items() if isinstance(value, RigidObjectCfg)]
    materials = resolve_materials(spec, get_usd_objects_info(scene_path) if spec.active else [],
                                  object_names, catalog if catalog is not None else load_catalog())
    pad = pad_material(spec)
    if spec.active and pad is not None and not pad_bodies:
        print(f"[friction] WARNING: {type(env_cfg.scene).__name__}'s robot declares no "
              "friction_bodies label -- objects get the override, the finger pads keep their "
              "authored material (see docs/physics.md).")
    env_cfg.friction = provenance(spec, materials, pad, pad_bodies)
    for name, term in event_terms(materials, pad, pad_bodies).items():
        setattr(env_cfg.events, name, term)
    if spec.active:
        print(f"[friction] {spec.text}: {len(materials)} object(s)"
              + (f", pads {pad.static}/{pad.dynamic} on {list(pad_bodies)}" if pad and pad_bodies else ""))
    return env_cfg.friction


# --------------------------------------------------------------------------- #
# Readback: what PhysX actually holds after start-up
# --------------------------------------------------------------------------- #

def _shape_rows(mats) -> list[list[float]]:
    return [[round(float(v), 4) for v in row] for row in mats]


def summarise_rows(rows) -> dict:
    """One line per object for a human: the coefficient pair, the shape count, and
    whether every shape agrees."""
    if not isinstance(rows, list) or not rows:
        return {"shapes": 0}
    distinct = sorted({(round(r[0], 4), round(r[1], 4), round(r[2], 4)) for r in rows})
    s, d, r = distinct[0]
    return {"static": s, "dynamic": d, "restitution": r, "shapes": len(rows), "uniform": len(distinct) == 1,
            **({"distinct": [list(x) for x in distinct]} if len(distinct) > 1 else {})}


def read_applied(env, requested: dict | None = None) -> dict:
    """Read every rigid object's (and the pads') PhysX material properties, env 0.

    Returns ``{"objects": {name: [[static, dynamic, restitution] per shape]},
    "gripper": {body: [...]}, "requested": <provenance>}``. Works under ``upstream`` too.
    """
    from isaaclab.assets import Articulation, RigidObject

    requested = requested if requested is not None else getattr(env.cfg, "friction", None) or {}
    out = {"objects": {}, "gripper": {}, "requested": requested}
    for name, asset in env.scene.rigid_objects.items():
        if isinstance(asset, RigidObject):
            out["objects"][name] = _shape_rows(asset.root_physx_view.get_material_properties()[0])
    robot = env.scene.articulations.get("robot")
    bodies = requested.get("gripper_bodies") or []
    if isinstance(robot, Articulation) and bodies:
        mats = robot.root_physx_view.get_material_properties()[0]
        shapes_per_link = [robot._physics_sim_view.create_rigid_body_view(p).max_shapes
                           for p in robot.root_physx_view.link_paths[0]]
        for body in bodies:
            if body not in robot.body_names:
                out["gripper"][body] = "NOT A BODY"
                continue
            i = robot.body_names.index(body)
            start = sum(shapes_per_link[:i])
            out["gripper"][body] = _shape_rows(mats[start:start + shapes_per_link[i]])
    out["summary"] = {"objects": {n: summarise_rows(r) for n, r in out["objects"].items()},
                      "gripper": {b: summarise_rows(r) for b, r in out["gripper"].items()}}
    return out


def write_applied(env, output_dir: str) -> str:
    path = os.path.join(output_dir, APPLIED_FILENAME)
    with open(path, "w") as f:
        json.dump(read_applied(env), f, indent=1)
    return path


def check_applied(applied: dict, tol: float = 1e-3) -> list[str]:
    """Compare a ``friction_applied.json`` payload against its own ``requested`` block.
    Returns a list of mismatch descriptions (empty = everything landed). Pure; used by
    ``scripts/verify_patches.py``."""
    req = applied.get("requested") or {}
    problems = []
    for name, mat in (req.get("objects") or {}).items():
        rows = applied.get("objects", {}).get(name)
        if not rows:
            problems.append(f"{name}: no PhysX readback")
            continue
        for s, d, _r in rows:
            if abs(s - mat["static"]) > tol or abs(d - mat["dynamic"]) > tol:
                problems.append(f"{name}: requested {mat['static']}/{mat['dynamic']}, PhysX holds {s}/{d}")
                break
    pad = req.get("gripper")
    for body in req.get("gripper_bodies") or []:
        rows = applied.get("gripper", {}).get(body)
        if not isinstance(rows, list) or not rows:
            problems.append(f"pad {body}: no PhysX readback ({rows})")
            continue
        if pad:
            for s, d, _r in rows:
                if abs(s - pad["static"]) > tol or abs(d - pad["dynamic"]) > tol:
                    problems.append(f"pad {body}: requested {pad['static']}/{pad['dynamic']}, PhysX holds {s}/{d}")
                    break
    return problems
