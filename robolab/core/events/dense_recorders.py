# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense-annotation recorder terms (docs/verified/dense_annotations.md, §5 R2 and R4).

Three per-step channels that the event logic already computes and never wrote to
disk, so that the phase annotator (``robolab/eval/phases.py``) reads recorded facts
instead of re-deriving them:

* ``tracker/<object>`` (T, 2) uint8: ``GraspTracker``'s ``grasped`` and
  ``attempt_closed`` per object, as the live tracker held them on that step. The
  offline replay of the tracker (``robolab/eval/tracker_replay.py``) reproduced
  93.5 % of logged lines because the recording lacks the finger body it reads;
  with the state itself on disk there is nothing to replay.
* ``conditions/s<i>`` and ``conditions/s<i>_<group>_<j>`` (T,) uint8: every
  subtask condition of the task's ladder evaluated on every step, plus the
  per-subtask aggregate: each group's *terminal* condition (its rungs are
  sequential; ``grabbed`` is false again once the object is placed), combined
  under the subtask's ``logical`` mode. The legend (which
  predicate, which arguments) is written once to ``conditions_legend.json`` next
  to ``env_cfg.json``. ``gt_state.py`` evaluated the same thing under
  ``--enable-gt-state`` and never persisted it.
* ``contact_body/<label>__<object>`` (T,) float32: contact force magnitude of a
  robot body beyond the two finger pads (knuckles, outer fingers, wrist) against
  each object of ``contact_object_list``, from the batch sensors that
  ``create_contact_sensors`` adds for the robot's ``contact_extra_bodies`` label.
  A bin pushed 6.5 cm by the forearm has no robot contact on record without it.

Every term returns ``(None, None)`` when its source is missing, so a robot or task
without the label records exactly what it did before.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable

import torch
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Pure helpers (no simulator; covered by offline_tests/test_dense_recorders.py)
# ----------------------------------------------------------------------------

def _func_name(func: Callable) -> str:
    inner = getattr(func, "func", func)
    return getattr(inner, "__name__", repr(inner))


def _func_params(func: Callable) -> dict:
    kw = getattr(func, "keywords", None) or {}
    return {k: (v if isinstance(v, (int, float, str, bool, type(None))) else repr(v)) for k, v in kw.items()}


def condition_keys(subtasks) -> list[tuple[str, int, str | None, int | None]]:
    """(key, subtask index, group, condition index) in recording order; the
    aggregate row of each subtask has group None."""
    keys = []
    for i, st in enumerate(subtasks):
        keys.append((f"s{i}", i, None, None))
        for g, cond_list in st.conditions.items():
            for j in range(len(cond_list)):
                keys.append((f"s{i}_{g}_{j}", i, g, j))
    return keys


def condition_legend(subtasks) -> list[dict]:
    """One row per recorded ``conditions/`` dataset, in recording order."""
    rows = []
    for i, st in enumerate(subtasks):
        rows.append({"key": f"s{i}", "subtask": i, "name": getattr(st, "name", None),
                     "logical": getattr(st, "logical", "all"), "k": getattr(st, "K", None),
                     "group": None, "index": None, "func": None, "params": None, "score": None})
        for g, cond_list in st.conditions.items():
            for j, (func, score) in enumerate(cond_list):
                rows.append({"key": f"s{i}_{g}_{j}", "subtask": i, "name": getattr(st, "name", None),
                             "logical": getattr(st, "logical", "all"), "k": getattr(st, "K", None),
                             "group": g, "index": j, "func": _func_name(func),
                             "params": _func_params(func), "score": score})
    return rows


def aggregate_subtask(group_results: list[bool], logical: str, k: int | None) -> bool:
    """The subtask's own completion rule over its groups (mirrors gt_state)."""
    n = sum(bool(r) for r in group_results)
    if logical == "any":
        return n >= 1
    if logical == "choose":
        return n >= (k or 1)
    return all(bool(r) for r in group_results) if group_results else False


def evaluate_conditions(subtasks, num_envs: int, call: Callable[[Callable, int], bool]) -> dict[str, list[bool]]:
    """Evaluate every condition for every env. ``call(func, env_id) -> bool``;
    a raising condition reads False. Returns key -> per-env list, keys as in
    :func:`condition_keys`."""
    out: dict[str, list[bool]] = {}
    for i, st in enumerate(subtasks):
        per_env_groups: list[list[bool]] = [[] for _ in range(num_envs)]
        for g, cond_list in st.conditions.items():
            # A group's conditions are sequential rungs (grabbed -> in container); once the
            # object is placed the earlier rung is false again, so "the group holds now" is
            # its terminal condition, as gt_state's object_completed reads past subtasks.
            per_env_group_ok = [False] * num_envs
            for j, (func, _score) in enumerate(cond_list):
                vals = []
                for eid in range(num_envs):
                    try:
                        v = bool(call(func, eid))
                    except Exception as err:  # noqa: BLE001
                        logger.debug("condition %s (env %d) raised: %s", _func_name(func), eid, err)
                        v = False
                    vals.append(v)
                    per_env_group_ok[eid] = v
                out[f"s{i}_{g}_{j}"] = vals
            for eid in range(num_envs):
                per_env_groups[eid].append(per_env_group_ok[eid])
        out[f"s{i}"] = [aggregate_subtask(per_env_groups[eid], getattr(st, "logical", "all"), getattr(st, "K", None))
                        for eid in range(num_envs)]
    return out


def pack_tracker_state(pairs: dict[tuple[str, str], Any]) -> dict[str, torch.Tensor]:
    """``{(obj, hand): _PairState}`` -> ``{key: (N, 2) uint8}`` with columns
    [grasped, attempt_closed]; key is the object for the ``gripper`` hand and
    ``<obj>__<hand>`` otherwise."""
    out = {}
    for (obj, hand), st in pairs.items():
        key = obj if hand == "gripper" else f"{obj}__{hand}"
        out[key] = torch.stack([st.grasped, st.attempt_closed], dim=-1).to(torch.uint8)
    return out


def body_force_magnitudes(force_matrix_w: torch.Tensor, object_names: list[str]) -> dict[str, torch.Tensor]:
    """``force_matrix_w`` (N, B, M, 3) of one batch sensor -> ``{obj: (N,) float32}``,
    force magnitude summed over the sensor's bodies (B is 1 for a single link)."""
    if force_matrix_w.ndim != 4 or force_matrix_w.shape[2] != len(object_names):
        raise ValueError(f"force_matrix_w shape {tuple(force_matrix_w.shape)} does not match {len(object_names)} objects")
    mag = torch.linalg.norm(force_matrix_w, dim=-1).sum(dim=1)     # (N, M)
    return {obj: mag[:, m].to(torch.float32) for m, obj in enumerate(object_names)}


# ----------------------------------------------------------------------------
# Recorder terms
# ----------------------------------------------------------------------------

class PostStepTrackerStateRecorder(RecorderTerm):
    """R4a: the live ``GraspTracker`` per-object state, each step."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._announced = False

    def record_post_step(self):
        from robolab.core.task.grasp import _TRACKERS  # noqa: PLC0415
        try:
            tracker = _TRACKERS.get(self._env)
        except TypeError:
            tracker = getattr(self._env, "_grasp_tracker", None)
        if tracker is None or not getattr(tracker, "_pairs", None):
            return None, None
        if not self._announced:
            self._announced = True
            try:
                robot = self._env.scene["robot"]
                print(f"[dense] robot bodies: {list(robot.data.body_names)}")
                print(f"[dense] robot joints: {list(robot.data.joint_names)}")
            except Exception:  # noqa: BLE001
                pass
        return "tracker", pack_tracker_state(tracker._pairs)


class PostStepConditionsRecorder(RecorderTerm):
    """R4b: every ladder condition, every step, plus the per-subtask aggregate."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._subtasks = None
        self._initialized = False
        self._legend_written = False

    def _init(self):
        self._initialized = True
        from robolab.core.events.subtask_recorder import SubtaskCompletionRecorderTerm  # noqa: PLC0415
        rm = getattr(self._env, "recorder_manager", None)
        term = rm.get_term(SubtaskCompletionRecorderTerm) if rm is not None and hasattr(rm, "get_term") else None
        sms = getattr(term, "subtask_state_machines", None) if term is not None else None
        if not sms:
            logger.info("conditions recorder: no subtask state machine; channel disabled")
            return
        self._subtasks = list(sms[0].subtasks)

    def _write_legend(self):
        self._legend_written = True
        out_dir = getattr(self._env, "output_dir", None)
        if not out_dir:
            return
        try:
            with open(os.path.join(out_dir, "conditions_legend.json"), "w") as f:
                json.dump({"schema_version": 1, "conditions": condition_legend(self._subtasks)}, f, indent=1, default=str)
        except Exception:  # noqa: BLE001
            logger.exception("conditions recorder: legend not written")

    def record_post_step(self):
        if not self._initialized:
            self._init()
        if not self._subtasks:
            return None, None
        if not self._legend_written:
            self._write_legend()
        env = self._env
        n = env.num_envs
        res = evaluate_conditions(self._subtasks, n, lambda func, eid: func(env=env, env_id=eid))
        return "conditions", {k: torch.tensor(v, dtype=torch.uint8, device=env.device) for k, v in res.items()}


class PostStepExtraContactRecorder(RecorderTerm):
    """R2: force magnitude of each extra robot body against each object."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._initialized = False
        self._sensors: dict[str, Any] = {}
        self._objects: list[str] = []

    def _init(self):
        self._initialized = True
        extra = getattr(self._env.cfg, "contact_extra_bodies", None) or {}
        self._objects = list(getattr(self._env.cfg, "contact_object_list", None) or [])
        for label in extra:
            sensor = self._env.scene.sensors.get(f"{label}__all_objs")
            if sensor is None:
                logger.info("extra contact recorder: no sensor for %s", label)
                continue
            self._sensors[label] = sensor

    def record_post_step(self):
        if not self._initialized:
            self._init()
        if not self._sensors:
            return None, None
        out = {}
        for label, sensor in self._sensors.items():
            fm = sensor.data.force_matrix_w
            if fm is None:
                continue
            try:
                for obj, mag in body_force_magnitudes(fm, self._objects).items():
                    out[f"{label}__{obj}"] = mag
            except ValueError as err:
                logger.warning("extra contact recorder (%s): %s", label, err)
        return ("contact_body", out) if out else (None, None)


@configclass
class PostStepTrackerStateRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PostStepTrackerStateRecorder


@configclass
class PostStepConditionsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PostStepConditionsRecorder


@configclass
class PostStepExtraContactRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PostStepExtraContactRecorder
