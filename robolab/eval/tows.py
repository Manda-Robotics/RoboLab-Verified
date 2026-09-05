# SPDX-License-Identifier: Apache-2.0
"""Towing artifact detection from recorded state (docs/verified/towing.md, P124).

A "tow" is the object that rides on one finger pad with the hand open (or closed on nothing),
the "magnetic object" of the human reviews. The mechanism: an object feature pressed a few
millimetres into the Robotiq pad collider, a 6 mm slab with air behind it, is depenetrated out of
the pad's back face and hooked behind it. It happens when the finger cannot yield, i.e. when
``finger_joint`` sits at a hard limit (0.000 open, 0.785 closed). The signal is therefore the
depth of the object inside the pad's collision volume, which this module recomputes offline:

    object collision points (scene USD)  ->  world (recorded root pose)
      ->  finger frame (forward kinematics from the 13 recorded joint angles)
      ->  signed distance to the pad's bounding box (mm, negative = inside)

A *hook* is one pad at least ``TOW_HOOK_DEPTH_MM`` deep for ``TOW_HOOK_MIN_S`` while the other pad
is not touching. Hooks are classed by what the object does meanwhile: ``tow`` (rises
``TOW_RISE_M``), ``drag`` (moves ``TOW_DRAG_PATH_M`` along its support), ``press`` (neither: the arm
pushing the pad into a pinned object). A deep one-sided contact with a mid-range finger and the
far pad touching is a ``squeeze`` (a real pinch rendered with interpenetration) and is not a hook.
Tows are tiered: **A** (finger at a limit for most of the segment, median depth >= 3 mm, >= 2 s,
rise >= 3 cm) is the flag that marks an episode ``physics_artifact``; **B** is a shorter or
shallower hook at a limit; **C** is the rest.

No simulator. Needs h5py; the object meshes need ``pxr`` (usd-core, shipped with Isaac Sim and
``pip install usd-core``). Only the DROID Franka + Robotiq 2F-85 rig is supported; other robots
get an ``unsupported`` document.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

import robolab.constants
from robolab.constants import ASSET_DIR

# --- Robotiq 2F-85 kinematics, read from assets/robots/franka_robotiq_2f_85_flattened.usd -------
# Each joint: frame in body0 (p0, q0 wxyz) and in body1 (p1, q1); a revolute joint rotates about the
# joint frame's Z. child_pose = parent_pose * T(p0,q0) * Rz(theta) * T(p1,q1)^-1. The fixed joints
# (outer knuckle -> outer finger) are identities. Validated against recorded finger poses: 0.00 mm.
_JOINTS = {
    "finger_joint": dict(p0=[0.054662514, 0.030601144, 0.0], q0=[0.0, 0.707106769, 0.707106769, 0.0],
                         p1=[0.054662514, 0.030601144, 0.0], q1=[0.0, 0.707106769, 0.707106769, 0.0],
                         body0="base_link", body1="left_outer_knuckle"),
    "right_outer_knuckle_joint": dict(p0=[0.054662514, -0.030601144, 0.0], q0=[0.707106769, 0.0, 0.0, 0.707106769],
                                      p1=[0.054662514, -0.030601144, 0.0], q1=[0.707106769, 0.0, 0.0, 0.707106769],
                                      body0="base_link", body1="right_outer_knuckle"),
    "left_outer_finger_fixed": dict(p0=[0.0, 0.0, 0.0], q0=[1.0, 0.0, 0.0, 0.0], p1=[0.0, 0.0, 0.0], q1=[1.0, 0.0, 0.0, 0.0],
                                    body0="left_outer_knuckle", body1="left_outer_finger"),
    "right_outer_finger_fixed": dict(p0=[0.0, 0.0, 0.0], q0=[1.0, 0.0, 0.0, 0.0], p1=[0.0, 0.0, 0.0], q1=[1.0, 0.0, 0.0, 0.0],
                                     body0="right_outer_knuckle", body1="right_outer_finger"),
    "left_inner_finger_joint": dict(p0=[0.098085366, 0.067758501, -0.0133], q0=[0.0, -0.698406577, 0.715701222, 0.0],
                                    p1=[0.098085366, 0.067758501, -0.0133], q1=[0.0, -0.698406577, 0.715701222, 0.0],
                                    body0="left_outer_finger", body1="left_inner_finger"),
    "right_inner_finger_joint": dict(p0=[0.098085366, -0.067758501, -0.0133], q0=[0.0, 0.698406577, 0.715701222, 0.0],
                                     p1=[0.098085366, -0.067758501, -0.0133], q1=[0.0, 0.698406577, 0.715701222, 0.0],
                                     body0="right_outer_finger", body1="right_inner_finger"),
    "left_inner_finger_knuckle_joint": dict(p0=[0.104600057, 0.049857065, 0.0], q0=[0.707106948, 0.0, 0.0, -0.70710659],
                                            p1=[0.104600057, 0.049857065, 0.0], q1=[0.707106948, 0.0, 0.0, -0.70710659],
                                            body0="left_inner_finger", body1="left_inner_knuckle"),
    "right_inner_finger_knuckle_joint": dict(p0=[0.104600057, -0.049857065, 0.0], q0=[0.0, 0.707106948, 0.70710659, 0.0],
                                             p1=[0.104600057, -0.049857065, 0.0], q1=[0.0, 0.707106948, 0.70710665, 0.0],
                                             body0="right_inner_finger", body1="right_inner_knuckle"),
}
# Body order of the serial chains (each child's joint is looked up in _JOINTS by body1).
_CHAIN = ["left_outer_knuckle", "left_outer_finger", "left_inner_finger", "left_inner_knuckle",
          "right_outer_knuckle", "right_outer_finger", "right_inner_finger", "right_inner_knuckle"]
_JOINT_OF_BODY = {v["body1"]: k for k, v in _JOINTS.items()}
DEFAULT_JOINT_NAMES = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6",
                       "panda_joint7", "finger_joint", "right_outer_knuckle_joint", "left_inner_finger_joint",
                       "right_inner_finger_joint", "left_inner_finger_knuckle_joint", "right_inner_finger_knuckle_joint"]
FINGER_JOINT = "finger_joint"
FINGER_CLOSED = float(np.pi / 4)                 # DROID close command; the joint's upper limit

# The pad collider (Defeatured_2F_85_PAD_OPEN_fingertipsstep_01) as its axis-aligned box in the
# inner-finger frame, metres: x along the finger, y across the jaw (the pad face at |y| = 42.5 mm
# when open), z the pad width. The mesh itself is a 6 mm slab at the tip, see towing.md.
PAD_BOX = {
    "left": (np.array([0.111099732, 0.04249759, -0.011]), np.array([0.149099919, 0.061656776, 0.011])),
    "right": (np.array([0.111099732, -0.061656773, -0.011]), np.array([0.149099919, -0.042497587, 0.011])),
}
PAD_BODY = {"left": "left_inner_finger", "right": "right_inner_finger"}


# --- quaternion helpers (wxyz) ------------------------------------------------------------------
def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _qrot(q, v):
    return quat_to_R(q) @ v


def _qinv(q):
    return np.asarray(q) * np.array([1.0, -1.0, -1.0, -1.0])


def _compose(T1, T2):
    return (T1[0] + _qrot(T1[1], T2[0]), _qmul(T1[1], T2[1]))


def _inv(T):
    qi = _qinv(T[1])
    return (-_qrot(qi, T[0]), qi)


def _qz(th):
    return np.array([np.cos(th / 2), 0.0, 0.0, np.sin(th / 2)])


def gripper_link_poses(base_pos, base_quat, joint_pos, joint_names=None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """World pose ``(p, q_wxyz)`` of every Robotiq link from the ``base_link`` pose (the recorded
    ``ee_pose``) and the recorded joint vector. Joints missing from ``joint_names`` are taken as 0."""
    names = list(joint_names) if joint_names is not None else DEFAULT_JOINT_NAMES
    q = {n: float(v) for n, v in zip(names, joint_pos)}
    T = {"base_link": (np.asarray(base_pos, float), np.asarray(base_quat, float))}
    for body in _CHAIN:
        jn = _JOINT_OF_BODY[body]
        j = _JOINTS[jn]
        T0 = (np.array(j["p0"]), np.array(j["q0"]))
        T1 = (np.array(j["p1"]), np.array(j["q1"]))
        th = q.get(jn, 0.0)
        Tj = _compose(_compose(T0, (np.zeros(3), _qz(th))), _inv(T1))
        T[body] = _compose(T[j["body0"]], Tj)
    return T


def pad_signed_distance(side: str, points_world: np.ndarray, link_poses: dict) -> tuple[float, int]:
    """Min over the object's points of the signed distance to the pad box (mm; negative = inside),
    and how many points are inside."""
    p, q = link_poses[PAD_BODY[side]]
    Pl = (points_world - p) @ quat_to_R(q)               # world -> finger frame (R^T)
    lo, hi = PAD_BOX[side]
    c, h = (lo + hi) / 2, (hi - lo) / 2
    s = (np.abs(Pl - c) - h).max(1)
    return float(s.min() * 1000.0), int((s < 0).sum())


# --- object collision points from the scene USD --------------------------------------------------
_SCENE_CACHE: dict[str, dict[str, np.ndarray]] = {}


def _reroot(path: str) -> str:
    marker = f"{os.sep}assets{os.sep}"
    if marker in path:
        return os.path.join(ASSET_DIR, path.split(marker, 1)[1])
    return path


def _sample_surface(V: np.ndarray, F: np.ndarray, n: int, rng) -> np.ndarray:
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    if area.sum() <= 0:
        return np.zeros((0, 3))
    idx = rng.choice(len(F), n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    return (1 - s)[:, None] * a[idx] + (s * (1 - r2))[:, None] * b[idx] + (s * r2)[:, None] * c[idx]


def scene_collision_points(scene_path: str, n_points: int | None = None) -> dict[str, np.ndarray]:
    """``{object name: (N, 3) points}`` on each rigid body's collision meshes, in the body's root frame
    (rotation + translation only: an authored body scale stays on the points, bins are authored in cm).
    Surface-sampled plus the mesh vertices, so thin rims and edges are represented."""
    scene_path = _reroot(scene_path)
    n_points = n_points or robolab.constants.TOW_OBJECT_POINTS
    key = f"{scene_path}|{n_points}"
    if key in _SCENE_CACHE:
        return _SCENE_CACHE[key]
    from pxr import Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

    def mat(m):
        return np.array([[m[i][j] for j in range(4)] for i in range(4)]).T

    rng = np.random.default_rng(0)
    out: dict[str, np.ndarray] = {}
    stage = Usd.Stage.Open(scene_path)
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    for root in stage.Traverse():
        if not root.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        Mb = mat(xc.GetLocalToWorldTransform(root))
        sc = np.linalg.norm(Mb[:3, :3], axis=0)
        Rb, tb = Mb[:3, :3] / sc, Mb[:3, 3]
        V, F, off = [], [], 0
        for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if p == root or not p.HasAPI(UsdPhysics.CollisionAPI):
                continue
            for m in Usd.PrimRange(p, Usd.TraverseInstanceProxies()):
                if not m.IsA(UsdGeom.Mesh):
                    continue
                mesh = UsdGeom.Mesh(m)
                pts = mesh.GetPointsAttr().Get()
                if not pts:
                    continue
                P = np.array(pts, float)
                Mm = mat(xc.GetLocalToWorldTransform(m))
                P = ((Mm[:3, :3] @ P.T).T + Mm[:3, 3] - tb) @ Rb
                fc, fi = mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get()
                if fc is not None and fi is not None:
                    fc, fi = np.asarray(fc), np.asarray(fi)
                    tris, k = [], 0
                    for c in fc:
                        idx = fi[k:k + c] + off
                        k += c
                        for j in range(1, c - 1):
                            tris.append((idx[0], idx[j], idx[j + 1]))
                    if tris:
                        F.append(np.array(tris, dtype=np.int64))
                V.append(P)
                off += len(P)
        if not V:
            continue
        V = np.concatenate(V)
        pts = V
        if F:
            F = np.concatenate(F)
            F = F[(F < len(V)).all(1)]
            pts = np.concatenate([V, _sample_surface(V, F, n_points, rng)])
        if len(pts) > 2 * n_points:
            pts = pts[rng.choice(len(pts), 2 * n_points, replace=False)]
        out[root.GetName()] = pts
    _SCENE_CACHE[key] = out
    return out


# --- episode analysis -----------------------------------------------------------------------------
@dataclass
class Segment:
    object: str
    pad: str
    t_start: float
    t_end: float
    duration_s: float
    depth_med_mm: float
    depth_max_mm: float
    far_pad_med_mm: float
    finger_min: float
    finger_max: float
    limit_frac: float
    rise_m: float
    path_m: float
    cls: str
    tier: str | None

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def _read_env_cfg(task_dir: str) -> dict:
    with open(os.path.join(task_dir, "env_cfg.json")) as f:
        return json.load(f)


def _control_dt(cfg: dict) -> float:
    return float(cfg["sim"]["dt"]) * int(cfg.get("decimation", 1))


def classify(rows: list[list[float]], step_s: float, motion) -> list[Segment]:
    """``rows``: ``[t, sd_left_mm, sd_right_mm, finger_joint]`` at successive evaluated frames (only
    frames where a pad is within 0.5 mm need to be present). ``motion(t0, t1) -> (rise_m, path_m)``
    reads the object's motion over a window. Returns the hook segments with class and tier."""
    C = robolab.constants
    segs, cur = [], None

    def close(c):
        dur = c["t1"] - c["t0"] + step_s
        if dur < C.TOW_HOOK_MIN_S:
            return
        fj = np.array(c["fj"])
        limit_frac = float(((fj < 0.03) | (fj > FINGER_CLOSED - 0.03)).mean())
        far_med = float(np.median(c["far"]))
        dmed = float(np.median(c["d"]))
        rise, path = motion(c["t0"], c["t1"] + step_s)
        at_limit = limit_frac >= 0.5
        if not at_limit and far_med < C.TOW_SQUEEZE_FAR_MM:
            cls = "squeeze"
        elif rise >= C.TOW_RISE_M:
            cls = "tow"
        elif path >= C.TOW_DRAG_PATH_M:
            cls = "drag"
        else:
            cls = "press"
        tier = None
        if cls == "tow":
            if at_limit and dmed >= C.TOW_TIER_A_DEPTH_MM and dur >= C.TOW_TIER_A_MIN_S and rise >= C.TOW_TIER_A_RISE_M:
                tier = "A"
            elif at_limit and dmed >= C.TOW_TIER_B_DEPTH_MM and dur >= C.TOW_TIER_B_MIN_S:
                tier = "B"
            else:
                tier = "C"
        segs.append(Segment(c["obj"], c["pad"], round(c["t0"], 3), round(c["t1"] + step_s, 3), round(dur, 3), dmed,
                            float(max(c["d"])), far_med, float(fj.min()), float(fj.max()), limit_frac, rise, path, cls, tier))

    for t, sL, sR, fj in rows:
        deep = None
        if sL <= -C.TOW_HOOK_DEPTH_MM and sL <= sR:
            deep = ("left", -sL, sR)
        elif sR <= -C.TOW_HOOK_DEPTH_MM:
            deep = ("right", -sR, sL)
        if deep and cur and cur["pad"] == deep[0] and t - cur["t1"] <= step_s * 1.5 + 1e-6:
            cur["t1"] = t
            cur["d"].append(deep[1])
            cur["far"].append(deep[2])
            cur["fj"].append(fj)
        elif deep:
            if cur:
                close(cur)
            cur = dict(pad=deep[0], t0=t, t1=t, d=[deep[1]], far=[deep[2]], fj=[fj], obj=None)
        else:
            if cur:
                close(cur)
            cur = None
    if cur:
        close(cur)
    return segs


def analyse_episode(task_dir: str, env_id: int, run_index: int = 0, step: int | None = None) -> dict:
    """Pad-penetration analysis of one recorded episode. Returns the ``tows_*.json`` document."""
    import h5py  # noqa: PLC0415
    cfg = _read_env_cfg(task_dir)
    dt = _control_dt(cfg)
    step = step or robolab.constants.TOW_EVAL_STEP
    joint_names = cfg.get("robot_joint_names") or DEFAULT_JOINT_NAMES
    doc = {"version": 1, "task_dir": task_dir, "env_id": env_id, "run_index": run_index, "dt": dt,
           "objects_checked": [], "segments": [], "summary": _summary([])}
    if FINGER_JOINT not in joint_names or "left_inner_finger_joint" not in joint_names:
        doc["status"] = "unsupported robot"
        return doc
    scene_path = cfg["scene"]["scene"]["spawn"]["usd_path"]
    meshes = scene_collision_points(scene_path)
    with h5py.File(os.path.join(task_dir, f"run_{run_index}.hdf5"), "r") as f:
        d = f["data"][f"demo_{env_id}"]
        jp = d["states/articulation/robot/joint_position"][:]
        ee_p = d["ee_pose/position"][:]
        ee_q = d["ee_pose/orientation"][:]
        n = len(jp)
        R = np.array([quat_to_R(q) for q in ee_q])
        fj_all = jp[:, joint_names.index(FINGER_JOINT)]
        half = 0.052 - 0.06 * np.clip(fj_all, 0, FINGER_CLOSED) / FINGER_CLOSED   # pad centre offset, near-filter only
        padL = ee_p + R[:, :, 0] * 0.13 + R[:, :, 1] * half[:, None]
        padR = ee_p + R[:, :, 0] * 0.13 - R[:, :, 1] * half[:, None]
        fk_cache: dict[int, dict] = {}
        for obj in d["states/rigid_object"]:
            if obj == "table" or obj not in meshes:
                continue
            P = meshes[obj]
            rad = float(np.linalg.norm(P, axis=1).max())
            if rad > robolab.constants.TOW_MAX_OBJECT_RADIUS_M:
                continue
            op = d[f"states/rigid_object/{obj}/root_pose"][:]
            near = np.nonzero(np.minimum(np.linalg.norm(op[:, :3] - padL, axis=1),
                                         np.linalg.norm(op[:, :3] - padR, axis=1)) < rad + 0.025)[0][::step]
            if len(near) == 0:
                continue
            doc["objects_checked"].append(obj)
            rows = []
            for i in near:
                if i not in fk_cache:
                    fk_cache[i] = gripper_link_poses(ee_p[i], ee_q[i], jp[i], joint_names)
                Pw = (quat_to_R(op[i, 3:]) @ P.T).T + op[i, :3]
                sL, _ = pad_signed_distance("left", Pw, fk_cache[i])
                sR, _ = pad_signed_distance("right", Pw, fk_cache[i])
                if sL < 0.5 or sR < 0.5:
                    rows.append([i * dt, sL, sR, float(jp[i, joint_names.index(FINGER_JOINT)])])

            def motion(t0, t1, op=op):
                a, b = int(round(t0 / dt)), min(int(round(t1 / dt)) + 1, n)
                seg = op[a:b, :3]
                if len(seg) < 2:
                    return 0.0, 0.0
                return float(seg[:, 2].max() - seg[:, 2].min()), float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum())

            for s in classify(rows, step * dt, motion):
                s.object = obj
                doc["segments"].append(s.as_dict())
    doc["segments"].sort(key=lambda s: s["t_start"])
    doc["summary"] = _summary(doc["segments"])
    return doc


def _summary(segments: list[dict]) -> dict:
    tows = [s for s in segments if s["cls"] == "tow"]
    order = {"A": 0, "B": 1, "C": 2}
    best = min((s["tier"] for s in tows if s["tier"]), key=lambda t: order[t], default=None)
    flagged = [s for s in tows if s["tier"] in ("A", "B")]
    return {
        "tow_tier": best,
        "n_tow": len(tows), "n_drag": sum(1 for s in segments if s["cls"] == "drag"),
        "n_press": sum(1 for s in segments if s["cls"] == "press"), "n_squeeze": sum(1 for s in segments if s["cls"] == "squeeze"),
        "towed_objects": sorted({s["object"] for s in flagged}),
        "tow_time_s": round(float(sum(s["duration_s"] for s in flagged)), 2),
        "first_tow_s": (float(min(s["t_start"] for s in flagged)) if flagged else None),
        "max_depth_mm": round(float(max((s["depth_max_mm"] for s in segments), default=0.0)), 1),
    }


def tows_path(task_dir: str, env_id: int, run_index: int = 0) -> str:
    return os.path.join(task_dir, f"tows_{run_index}_env{env_id}.json")


def write_tows(task_dir: str, env_id: int, run_index: int = 0) -> dict:
    """Analyse one episode and write ``tows_<run>_env<env>.json`` next to its log (the run path calls
    this after every episode when ``robolab.constants.FLAG_TOWS`` is on; ``scripts/flag_tows.py``
    calls it for existing runs). Returns the document."""
    doc = analyse_episode(task_dir, env_id, run_index)
    tmp = tows_path(task_dir, env_id, run_index) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f)
    os.replace(tmp, tows_path(task_dir, env_id, run_index))
    return doc
