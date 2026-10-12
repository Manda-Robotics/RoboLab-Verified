# SPDX-FileCopyrightText: Copyright (c) 2026 Manda Robotics
# SPDX-License-Identifier: Apache-2.0
"""CPU settling/perturbation probe using frozen exported Newton collision meshes.

Accept a full RoboLab model-effective.xml or an already isolated fixture. This
is a reduced CPU MuJoCo experiment, not a substitute for Newton/Kit GPU tests.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def isolate(source, target):
    root = ET.parse(source).getroot()
    world = root.find("worldbody")
    target_name = "_World_envs_env_0_scene_" + target
    keep = {target_name, "_World_envs_env_0_scene_table"}
    if target == "plasticpail_a02":
        keep.add("_World_envs_env_0_scene_large_storage_rack")
    for child in list(world):
        ground = child.tag == "geom" and "GroundPlane/CollisionPlane" in child.get("name", "")
        if child.get("name") not in keep and not ground:
            world.remove(child)
    if not any(b.get("name") == target_name for b in world.findall("body")):
        raise ValueError(f"Target body missing: {target_name}")
    for tag in ("contact", "equality", "actuator", "sensor", "keyframe", "tendon"):
        for child in root.findall(tag):
            root.remove(child)
    for body in world.findall("body"):
        if body.get("name") != target_name:
            for parent in body.iter():
                for child in list(parent):
                    if child.tag in ("joint", "freejoint"):
                        parent.remove(child)
    used = {g.get("mesh") for g in world.iter("geom") if g.get("mesh")}
    asset = root.find("asset")
    if asset is not None:
        for child in list(asset):
            if child.tag == "mesh" and child.get("name") not in used:
                asset.remove(child)
    return ET.tostring(root, encoding="unicode")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", choices=("utilityjug_a02", "plasticpail_a02"), default="utilityjug_a02")
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--condim", type=int, choices=(3, 4, 6))
    parser.add_argument("--rolling", type=float)
    parser.add_argument("--timeconst", type=float)
    parser.add_argument("--linear-nudge", type=float, default=0, help="Initial world-X velocity (m/s)")
    parser.add_argument("--angular-nudge", type=float, default=0, help="Initial body-Z angular velocity (rad/s)")
    parser.add_argument("--height", type=float, default=0, help="Initial vertical offset (m)")
    parser.add_argument("--start-state", type=Path, help="result.json from the same source and target")
    args = parser.parse_args()
    for key in ("seconds", "rolling", "timeconst", "linear_nudge", "angular_nudge", "height"):
        value = getattr(args, key)
        if value is not None and not np.isfinite(value):
            parser.error(f"--{key.replace('_', '-')} must be finite")
    if args.seconds < 5 or (args.rolling is not None and args.rolling < 0):
        parser.error("Require at least five seconds and nonnegative rolling resistance")
    if args.output.exists():
        parser.error("Use a fresh output path")
    source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
    xml = isolate(args.source, args.target)
    model = mujoco.MjModel.from_xml_string(xml)
    if (model.nq, model.nv) != (7, 6):
        raise ValueError("Fixture must have exactly one free dynamic body")
    if args.timeconst is not None:
        if args.timeconst < 2 * model.opt.timestep:
            parser.error("Contact time constant must be at least twice the timestep")
        model.geom_solref[:] = [args.timeconst, 1]
    if args.condim is not None:
        model.geom_condim[:] = args.condim
    if args.rolling is not None:
        model.geom_friction[:, 2] = args.rolling
    data = mujoco.MjData(model)
    if args.start_state:
        saved = json.loads(args.start_state.read_text())
        if saved["source_sha256"] != source_hash or saved["target"] != args.target:
            raise ValueError("Start state belongs to a different source/target")
        data.qpos[:] = saved["qpos"]
    data.qpos[2] += args.height
    data.qvel[:] = [args.linear_nudge, 0, 0, 0, 0, args.angular_nudge]
    args.output.mkdir(parents=True)
    (args.output / "isolated-source.xml").write_text(xml)
    rows = []
    steps = round(args.seconds / model.opt.timestep)
    stride = max(1, round(1 / 60 / model.opt.timestep))
    started = time.monotonic()
    for step in range(steps + 1):
        if step % stride == 0 or step == steps:
            rows.append([float(data.time), *data.qpos.tolist(), *data.qvel.tolist()])
        if step < steps:
            mujoco.mj_step(model, data)
    states = np.asarray(rows)
    np.save(args.output / "states.npy", states)
    if not np.isfinite(states).all() or np.any(data.warning.number):
        raise RuntimeError(f"Invalid rollout: warnings={data.warning.number.tolist()}")
    tail = states[states[:, 0] >= states[-1, 0] - 5]
    positions = tail[:, 1:3]
    span = np.linalg.norm(positions[:, None] - positions[None, :], axis=2).max()
    bid = int(np.flatnonzero(model.body_jntnum)[0])
    result = {"source_sha256": source_hash, "target": args.target, "engine": "CPU MuJoCo",
              "mujoco_version": mujoco.__version__, "wall_seconds": time.monotonic() - started,
              "seconds": float(data.time), "dt": model.opt.timestep,
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "mass_kg": float(model.body_mass[bid]), "local_com": model.body_ipos[bid].tolist(),
              "principal_inertia": model.body_inertia[bid].tolist(),
              "effective_condim": np.unique(model.geom_condim).tolist(),
              "effective_friction": np.unique(model.geom_friction, axis=0).tolist(),
              "effective_solref": np.unique(model.geom_solref, axis=0).tolist(),
              "tail_xy_span_mm": float(span * 1000),
              "tail_angular_rms": float(np.sqrt(np.mean(np.sum(tail[:, 11:14] ** 2, axis=1)))),
              "qpos": data.qpos.tolist(), "qvel": data.qvel.tolist(),
              "state_columns": "time, root_position_xyz, root_quaternion_wxyz, free_joint_velocity_6",
              "warning_counts": data.warning.number.tolist()}
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
