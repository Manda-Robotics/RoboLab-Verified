# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import logging
import torch

import robolab.constants

from robolab.core.task.conditionals import (
    get_wrong_object_grabbed,
    gripper_fully_closed,
    gripper_hit_table,
    object_grabbed,
    object_upright,
)
from robolab.core.task.predicate_logic import in_contact
from robolab.core.task.collateral import CollateralTracker
from robolab.core.task.status import StatusCode
from robolab.core.world.world_state import get_world


logger = logging.getLogger(__name__)


class EventTracker:
    """
    Tracks grasp-related events across multiple parallel environments.

    Uses batched WorldState queries (env_id=None) for efficient per-env event
    detection. All internal state is stored as (num_envs,) tensors.

    Events tracked:
    - WRONG_OBJECT_GRABBED: When gripper grabs an object not in the intended target list
    - GRIPPER_HIT_TABLE: When gripper makes contact with table
    - GRIPPER_FULLY_CLOSED: When gripper closes fully (potential failed grasp)
    - OBJECT_STARTED_MOVING: Non-target object transitioned from stationary to moving
    - OBJECT_BUMPED: When object stops after small movement (< move_threshold), minor collision
    - OBJECT_MOVED: When object stops after significant movement (>= move_threshold), knocked/pushed
    - OBJECT_OUT_OF_SCENE: Object moved outside the workspace bounding box (fell off table)
    - OBJECT_TIPPED_OVER: Object that should be upright has fallen over
    - TARGET_OBJECT_DROPPED: Target object was grabbed but dropped mid-transport
    - GRIPPER_HIT_OBJECT: Gripper collided with a non-target object
    - MULTIPLE_OBJECTS_GRABBED: Gripper is in contact with multiple objects simultaneously

    Each event is recorded only on first occurrence per env. The tracker resets when
    the condition clears, allowing the event to be recorded again if it reoccurs.
    """

    def __init__(
        self,
        num_envs: int = 1,
        device: torch.device = None,
        bump_threshold: float = 0.02,   # 5 cm missed real knocks (wooden_bowl nudged 2.8 cm, pod 2026-08-26)
        move_threshold: float = 0.50,
        velocity_threshold: float = 0.05,
        workspace_center: tuple[float, float, float] = (0.55, 0.0, 0.5),
        workspace_size: tuple[float, float, float] = (2.0, 2.0, 2.0)
    ):
        self.num_envs = num_envs
        self.device = device or torch.device("cpu")
        self.bump_threshold = bump_threshold
        self.move_threshold = move_threshold
        self.velocity_threshold = velocity_threshold

        self.workspace_center = torch.tensor(workspace_center, device=self.device)
        self.workspace_half_size = torch.tensor(workspace_size, device=self.device) / 2.0
        self.reset()

    def reset(self) -> None:
        """Reset all event trackers to initial state for all envs."""
        N, dev = self.num_envs, self.device

        # Per-env wrong object grab tracking (string names, must be dict)
        self._recorded_wrong_object_grab: dict[int, str | None] = {i: None for i in range(N)}

        # Per-env bool tensors
        self._recorded_gripper_hit_table = torch.zeros(N, dtype=torch.bool, device=dev)
        self._recorded_gripper_fully_closed = torch.zeros(N, dtype=torch.bool, device=dev)
        self._was_warm = torch.ones(N, dtype=torch.bool, device=dev)          # inside the reset warm-up window
        self._collateral = CollateralTracker(N, dev)                            # A1 / B7: non-targets entering goal containers
        self._settling: dict[int, dict[str, float]] = {}                     # env_id -> {object: displacement} seen during warm-up
        self._recorded_multiple_grab = torch.zeros(N, dtype=torch.bool, device=dev)
        self._target_was_grabbed = torch.zeros(N, dtype=torch.bool, device=dev)
        self._recorded_target_dropped = torch.zeros(N, dtype=torch.bool, device=dev)

        # Per-object per-env state (populated lazily)
        self._object_is_moving: dict[str, torch.Tensor] = {}          # obj -> (N,) bool
        self._position_when_started_moving: dict[str, torch.Tensor] = {}  # obj -> (N, 3)
        self._started_moving_mask: dict[str, torch.Tensor] = {}       # obj -> (N,) bool: which envs have a start pos
        self._recorded_out_of_scene: dict[str, torch.Tensor] = {}     # obj -> (N,) bool
        self._recorded_tipped_objects: dict[str, torch.Tensor] = {}   # obj -> (N,) bool
        self._recorded_gripper_hit_objects: dict[str, torch.Tensor] = {}  # obj -> (N,) bool

    def reset_envs(self, env_ids: list[int]) -> None:
        """Reset event state for specific envs only."""
        for eid in env_ids:
            self._recorded_wrong_object_grab[eid] = None
        idx = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        self._recorded_gripper_hit_table[idx] = False
        self._recorded_gripper_fully_closed[idx] = False
        self._was_warm[idx] = True
        self._collateral.reset_envs(idx.tolist() if hasattr(idx, "tolist") else list(idx))
        for i in (idx.tolist() if hasattr(idx, "tolist") else list(idx)):
            self._settling.pop(int(i), None)
        self._recorded_multiple_grab[idx] = False
        self._target_was_grabbed[idx] = False
        self._recorded_target_dropped[idx] = False
        for d in (self._object_is_moving, self._position_when_started_moving,
                  self._started_moving_mask, self._recorded_out_of_scene,
                  self._recorded_tipped_objects, self._recorded_gripper_hit_objects):
            for t in d.values():
                t[idx] = 0

    def _is_outside_workspace_batched(self, positions: torch.Tensor) -> torch.Tensor:
        """Check if positions are outside workspace. positions: (N, 3), returns (N,) bool."""
        diff = torch.abs(positions - self.workspace_center)
        return torch.any(diff > self.workspace_half_size, dim=-1)

    def _get_not_intended_mask(self, obj_name: str, per_env_intended: list[set[str]]) -> torch.Tensor:
        """Return (N,) bool mask: True where obj_name is NOT in that env's intended set."""
        return torch.tensor(
            [obj_name not in per_env_intended[eid] for eid in range(self.num_envs)],
            dtype=torch.bool, device=self.device
        )

    def check_events(
        self,
        env: Any,
        per_env_intended: list[set[str]],
        frozen_mask: torch.Tensor | None = None,
        ignore_objects: list[str] = None,
        upright_objects: list[str] = None,
        verbose: bool = False,
        per_env_allowed: list[set[str]] | None = None,
        per_env_containers: list[set[str]] | None = None,
    ) -> list[tuple[str, StatusCode, torch.Tensor]]:
        """
        Check for events across all envs using batched queries.

        Args:
            env: The environment object
            per_env_intended: Per-env sets of the current stage's target object names
                (drop tracking on, bump/move tracking off)
            per_env_allowed: Per-env sets of objects the policy may grasp without a
                WRONG_OBJECT_GRABBED — every stage's targets. Defaults to
                ``per_env_intended``.
            per_env_containers: Per-env destination objects; never reported as a
                wrong grab (the hand touches the bin while releasing into it).
            frozen_mask: (num_envs,) bool tensor, True for frozen envs to skip
            ignore_objects: Objects to ignore (default: ["table"])
            upright_objects: Objects that should remain upright
            verbose: Whether to print event messages

        Returns:
            List of (info_string, StatusCode, env_mask) where env_mask is (num_envs,) bool
            indicating which envs the event applies to.
        """
        events = []
        if frozen_mask is None:
            frozen_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        active_mask = ~frozen_mask

        if ignore_objects is None:
            ignore_objects = ["table"]
        ignore_set = set(ignore_objects)

        world = get_world(env)

        if per_env_allowed is None:
            per_env_allowed = per_env_intended
        if per_env_containers is None:
            per_env_containers = [set() for _ in range(self.num_envs)]

        # --- Grasp attempts / releases / drops (from the shared GraspTracker) ---
        from robolab.core.task.grasp import get_grasp_tracker
        tracker = get_grasp_tracker(env)
        for obj_name in sorted(set().union(*per_env_intended) | set().union(*per_env_allowed) if per_env_allowed else set().union(*per_env_intended)):
            try:
                tracker.update(obj_name, "gripper")
            except Exception:
                continue
        step_dt = float(getattr(env, "step_dt", 0.0) or 1 / 15)
        for eid, obj_name, hand, kind, extra in tracker.pop_events():
            if frozen_mask[eid]:
                continue
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            mask[eid] = True
            onset = extra.get("onset_step") if isinstance(extra, dict) else None
            is_container = bool(per_env_containers and obj_name in per_env_containers[eid])
            if kind == "gripped":
                # P78: the rung before the carry. A container is excluded for the same
                # reason P72 excludes it from attempts: closing on a bin rim is not a grip
                # anyone wants counted.
                if not is_container:
                    events.append((f"'{obj_name}' gripped (jaws closed on it)",
                                   StatusCode.OBJECT_GRIPPED, mask, onset))
            elif kind == "grabbed":
                # P71: the detector reports a physical carry; it is NOT the progress signal.
                # Emitting it as *_SUCCESS put a green "success" on 28 grasps of objects the
                # task then flagged as WRONG (The reviewer: "it's giving an object grab success flag
                # for the wrong object"). OBJECT_CARRIED is neutral; the ladder's
                # OBJECT_GRABBED_SUCCESS remains the green line that means progress.
                events.append((f"'{obj_name}' carried (grasp established)",
                               StatusCode.OBJECT_CARRIED, mask, onset))
            elif kind == "attempt_failed":
                n = int(extra.get("count", 1))
                span = (int(extra.get("last_step", 0)) - int(extra.get("first_step", 0))) * step_dt
                info = (f"Grasp attempt on '{obj_name}' failed (contact lost before a carry was established)"
                        if n <= 1 else
                        f"Grasp attempts on '{obj_name}' failed ×{n} over {span:.1f}s (contact never became a carry)")
                # P61: stamp the burst at the first attempt, not at the flush that
                # happens GRASP_ATTEMPT_BURST_S later
                # P72: a brush against a bin or shelf is not a grasp attempt. 14 of 81
                # attempt lines in rc3 were on a container or fixture (The reviewer: "I don't think
                # this was a grasp attempt on bin"). A genuine attempt on a container is
                # still reported by the wrong-object machinery.
                if not is_container:
                    events.append((info, StatusCode.GRASP_ATTEMPT_FAILED, mask,
                                   int(extra.get("first_step", 0)) or None))
            elif kind == "released":
                events.append((f"'{obj_name}' released (hand opened)", StatusCode.OBJECT_RELEASED, mask))
            elif kind == "towed":
                events.append((f"'{obj_name}' towed without a grasp — moving with an OPEN hand (stuck to a finger; physics artifact, episode is not trustworthy)", StatusCode.TOWED_WITHOUT_GRASP, mask))
            else:
                events.append((f"'{obj_name}' dropped (left the closed hand)", StatusCode.OBJECT_DROPPED, mask))
            if verbose:
                print(f"[EventTracker] env{eid}: {events[-1][0]}")

        # publish towed objects for the results row (episode is bogus if non-empty)
        try:
            env._towed_objects = {eid: sorted(v) for eid, v in tracker.towed_objects().items()}
        except Exception:
            pass

        # --- Off the table (P38): any object, one flag; TARGET_LOST when the task is unrecoverable ---
        try:
            events.extend(self._check_off_table_batched(env, ignore_set, active_mask, verbose))
        except Exception:
            logger.exception("off-table check failed")

        # --- Collateral placement: a non-target enters a goal container (A1, B7) ---
        try:
            events.extend(self._check_collateral_batched(env, tracker, per_env_allowed, per_env_containers, ignore_set, active_mask, verbose))
        except Exception:
            logger.exception("collateral placement check failed")

        # --- Wrong object grabbed (P44): a *carry* (GraspTracker) of an object that is
        # not a target. Upstream used contact + a closure gate, which fired on every
        # touch of a decoy (FruitsGreenLimesOnPlate cosmos3_s2 env 0: six flags on the
        # lemon in 13 s, five of them touches). ---
        scene_objs = [o for o in (getattr(env.cfg, "contact_object_list", None) or []) if o not in ignore_set]
        for eid in range(self.num_envs):
            if frozen_mask[eid]:
                continue
            wrong_obj = None
            for o in scene_objs:
                if o in per_env_allowed[eid] or o in per_env_containers[eid]:
                    continue
                try:
                    if tracker.grasped(o, "gripper", env_id=eid):
                        wrong_obj = o
                        break
                except Exception:
                    continue
            if wrong_obj is not None:
                if self._recorded_wrong_object_grab[eid] != wrong_obj:
                    info = f"Wrong object grabbed: '{wrong_obj}' (target objects: {sorted(per_env_allowed[eid])})"
                    mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                    mask[eid] = True
                    events.append((info, StatusCode.WRONG_OBJECT_GRABBED, mask))
                    self._recorded_wrong_object_grab[eid] = wrong_obj
                    if verbose:
                        print(f"[EventTracker] env{eid}: {info}")
            else:
                if self._recorded_wrong_object_grab[eid] is not None:
                    info = f"Wrong object that was grabbed is now detached: '{self._recorded_wrong_object_grab[eid]}'"
                    mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                    mask[eid] = True
                    events.append((info, StatusCode.WRONG_OBJECT_DETACHED, mask))
                    if verbose:
                        print(f"[EventTracker] env{eid}: {info}")
                self._recorded_wrong_object_grab[eid] = None

        # --- Gripper hit table (batched) ---
        hit_table = gripper_hit_table(env, env_id=None)  # (N,) bool
        new_hit = hit_table & ~self._recorded_gripper_hit_table & active_mask
        if new_hit.any():
            events.append(("Gripper hit table", StatusCode.GRIPPER_HIT_TABLE, new_hit.clone()))
            self._recorded_gripper_hit_table |= new_hit
            if verbose:
                envs = new_hit.nonzero(as_tuple=False).squeeze(-1).tolist()
                print(f"[EventTracker] envs {envs}: Gripper hit table")
        # Reset recording for envs where condition cleared
        cleared = ~hit_table & self._recorded_gripper_hit_table & active_mask
        self._recorded_gripper_hit_table &= ~cleared

        # --- Gripper fully closed ON NOTHING (batched) ---
        # Upstream flagged every full closure, including the one holding the
        # object (25 % of the corpus flags — findings.md B5, H-R5-9, H-R6-6).
        # Now: closed AND no scene object in contact with the hand = an air grasp.
        # P66: the event uses its own, strict threshold -- see
        # robolab.constants.GRIPPER_CLOSED_EVENT_THRESHOLD. The predicate's 0.75
        # default is shared with gripper_slightly_closed and is left alone.
        fully_closed = gripper_fully_closed(
            env, env_id=None,
            closed_threshold=float(getattr(robolab.constants, "GRIPPER_CLOSED_EVENT_THRESHOLD", 0.75)),
        )  # (N,) bool
        holding = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        hand_candidates = [o for o in (getattr(env.cfg, "contact_object_list", None) or []) if o not in ignore_set]
        for eid in fully_closed.nonzero(as_tuple=False).flatten().tolist():
            try:
                if world.get_objects_in_contact_with("gripper", hand_candidates, env_id=eid):
                    holding[eid] = True
            except Exception:
                pass
        closed_on_air = fully_closed & ~holding
        # Not during the reset warm-up: the hand's initial pose can read as closed for a
        # step or two before the policy has acted (seen at step 2 on MarkerInMug).
        _dt = float(getattr(env, "step_dt", 0.0) or 0.0)
        _warm = (env.episode_length_buf.float() * _dt) < float(getattr(robolab.constants, "SETTLE_WARMUP_S", 0.0) or 0.0)
        new_closed = closed_on_air & ~self._recorded_gripper_fully_closed & active_mask & ~_warm
        if new_closed.any():
            events.append(("Gripper closed on nothing", StatusCode.GRIPPER_FULLY_CLOSED, new_closed.clone()))
            self._recorded_gripper_fully_closed |= new_closed
            if verbose:
                envs = new_closed.nonzero(as_tuple=False).squeeze(-1).tolist()
                print(f"[EventTracker] envs {envs}: Gripper fully closed")
        cleared = ~fully_closed & self._recorded_gripper_fully_closed & active_mask
        self._recorded_gripper_fully_closed &= ~cleared

        # --- Reset warm-up: motion in the first SETTLE_WARMUP_S with the hand not
        # touching the object is the scene settling, not a robot bump (B12).
        step_dt = float(getattr(env, "step_dt", 0.0) or 0.0)
        warm = (env.episode_length_buf.float() * step_dt) < float(getattr(robolab.constants, "SETTLE_WARMUP_S", 0.0) or 0.0)
        just_left_warmup = self._was_warm & ~warm & active_mask
        for eid in just_left_warmup.nonzero(as_tuple=False).flatten().tolist():
            moved = self._settling.pop(eid, None)
            if moved:
                desc = ", ".join(f"{o} {d:.3f}m" for o, d in sorted(moved.items(), key=lambda kv: -kv[1]))
                mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device); mask[eid] = True
                events.append((f"Scene settling at reset (no contact with the hand): {desc}", StatusCode.SCENE_SETTLING, mask))
        self._was_warm = warm.clone()
        self._warm_now = warm

        # --- Movement transitions (batched per object) ---
        movement_events = self._check_movement_transitions_batched(
            env, per_env_intended, ignore_set, active_mask, verbose
        )
        events.extend(movement_events)

        # --- Out of scene (batched per object) ---
        out_events = self._check_out_of_scene_batched(
            env, per_env_intended, ignore_set, active_mask, verbose
        )
        events.extend(out_events)

        # --- Tipped objects (batched per object) ---
        if upright_objects:
            tipped_events = self._check_tipped_objects_batched(
                env, upright_objects, active_mask, verbose
            )
            events.extend(tipped_events)

        # --- Target dropped (batched) ---
        drop_events = self._check_target_dropped_batched(
            env, per_env_intended, active_mask, verbose
        )
        events.extend(drop_events)

        # --- Gripper-object collision (batched per object) ---
        collision_events = self._check_gripper_object_collision_batched(
            env, per_env_intended, ignore_set, active_mask, verbose
        )
        events.extend(collision_events)

        # --- Multiple objects grabbed (batched) ---
        multi_events = self._check_multiple_objects_grabbed_batched(
            env, ignore_set, active_mask, verbose
        )
        events.extend(multi_events)

        return events

    def _check_off_table_batched(self, env, ignore_set, active_mask, verbose):
        from robolab.core.task.off_table import get_monitor, required_groups, task_lost
        mon = get_monitor(env)
        objs = [o for o in (getattr(env.cfg, "contact_object_list", None) or []) if o not in ignore_set]
        fallen = mon.fallen_masks(objs)
        events = []
        for o, f in fallen.items():
            flagged = mon.flagged.setdefault(o, torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
            new = f & ~flagged & active_mask
            if new.any():
                events.append((f"'{o}' fell off the table", StatusCode.OBJECT_FELL_OFF_TABLE, new.clone()))
                flagged |= new
        try:
            groups = required_groups(dict(getattr(env.cfg.terminations.success, "params", {}) or {}))
        except Exception:
            groups = []
        if groups:
            lost = task_lost(fallen, groups, self.num_envs, self.device) & ~mon.lost_flagged & active_mask
            for eid in lost.nonzero(as_tuple=False).flatten().tolist():
                gone = [o for o, f in fallen.items() if bool(f[eid])]
                mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device); mask[eid] = True
                events.append((f"Target lost: {', '.join(gone)} off the table — the task can no longer succeed", StatusCode.TARGET_LOST, mask))
            mon.lost_flagged |= lost
        return events

    def _check_collateral_batched(self, env, tracker, per_env_allowed, per_env_containers, ignore_set, active_mask, verbose):
        """One flag per non-target object that enters a goal container after the
        warm-up: WRONG_OBJECT_PLACED if the hand held it recently, else
        WRONG_OBJECT_PUSHED_IN. Objects inside at reset never count."""
        from robolab.core.task.conditionals import object_in_container
        containers = sorted(set().union(*per_env_containers)) if per_env_containers else []
        if not containers:
            return []
        targets = set().union(*per_env_allowed) if per_env_allowed else set()
        scene_objs = list((getattr(env.cfg, "contact_object_list", None) or []))
        candidates = [o for o in scene_objs if o not in ignore_set and o not in targets and o not in containers]
        if not candidates:
            return []
        warm = getattr(self, "_warm_now", None)
        if warm is None:
            warm = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        inside = {}
        for c in containers:
            for o in candidates:
                try:
                    r = object_in_container(env, object=o, container=c, env_id=None)
                except Exception:
                    continue
                inside[(o, c)] = r if isinstance(r, torch.Tensor) else torch.as_tensor(r, dtype=torch.bool, device=self.device).reshape(-1)
        held = {o: tracker.recently_held(o, "gripper") for o in candidates}
        events = []
        for eid, obj, cont, kind in self._collateral.update(inside, warm, held):
            if not active_mask[eid]:
                continue
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device); mask[eid] = True
            # One flag either way (The reviewer 2026-08-26: "not sure the distinction is
            # necessary… would probably just have placed") — the wording still says
            # whether the hand was carrying it, since grasp detection can miss a carry.
            how = "released inside" if kind == "placed" else "ended up in"
            events.append((f"Wrong object placed: '{obj}' {how} '{cont}' (not a target)", StatusCode.WRONG_OBJECT_PLACED, mask))
            if verbose:
                print(f"[EventTracker] env{eid}: {events[-1][0]}")
        # publish the count for the results row
        env._collateral_placed = {eid: len(v) for eid, v in self._collateral.collateral.items()}
        return events

    def _check_movement_transitions_batched(
        self, env, per_env_intended, ignore_set, active_mask, verbose
    ) -> list[tuple[str, StatusCode, torch.Tensor]]:
        events = []
        world = get_world(env)

        objects_to_check = [
            obj for obj in world.objects.keys()
            if obj not in ignore_set
        ]

        # H-B17 / P51: movement is tracked for EVERY object, targets included — a
        # target knocked across the table used to produce no event at all. What is
        # suppressed instead is movement *the policy is doing on purpose*: while the
        # object is grasped, and briefly after it is released (the grasp/release
        # events already say that).
        from robolab.core.task.grasp import get_grasp_tracker
        grasp_tracker = get_grasp_tracker(env)
        for obj_name in objects_to_check:
            try:
                handled = grasp_tracker.grasped(obj_name, "gripper") | grasp_tracker.recently_held(obj_name, "gripper", within_s=1.0)
            except Exception:
                handled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            eligible = active_mask & ~handled

            if not eligible.any():
                continue

            try:
                current_pos, _ = world.get_pose(obj_name, env_id=None)  # (N, 3)
                velocity = world.get_velocity(obj_name, env_id=None)    # (N, 6)
                linear_speed = torch.norm(velocity[:, :3], dim=-1)       # (N,)
                is_moving = linear_speed > self.velocity_threshold

                was_moving = self._object_is_moving.get(
                    obj_name, torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                )

                # Started moving
                started = is_moving & ~was_moving & eligible
                if started.any():
                    if obj_name not in self._position_when_started_moving:
                        self._position_when_started_moving[obj_name] = torch.zeros(self.num_envs, 3, device=self.device)
                        self._started_moving_mask[obj_name] = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                    self._position_when_started_moving[obj_name][started] = current_pos[started]
                    self._started_moving_mask[obj_name] |= started
                    # Don't emit OBJECT_STARTED_MOVING as a separate event in the return;
                    # it's used internally for displacement tracking

                # Stopped moving
                stopped = ~is_moving & was_moving & eligible
                has_start = self._started_moving_mask.get(
                    obj_name, torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                )
                stopped_with_start = stopped & has_start

                if stopped_with_start.any():
                    start_pos = self._position_when_started_moving[obj_name]
                    displacement = torch.norm(current_pos - start_pos, dim=-1)  # (N,)

                    # warm-up diversion: not touched by the hand → settling, not a bump
                    warm_now = getattr(self, "_warm_now", None)
                    if warm_now is not None and bool((stopped_with_start & warm_now).any()):
                        try:
                            touched = in_contact(world, obj_name, "gripper", env_id=None)
                        except Exception:
                            touched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                        settling = stopped_with_start & warm_now & ~touched & (displacement >= self.bump_threshold)
                        for eid in settling.nonzero(as_tuple=False).flatten().tolist():
                            prev = self._settling.get(eid, {}).get(obj_name, 0.0)
                            self._settling.setdefault(eid, {})[obj_name] = max(prev, float(displacement[eid]))
                        stopped_with_start = stopped_with_start & ~settling
                    moved_mask = stopped_with_start & (displacement >= self.move_threshold)
                    if moved_mask.any():
                        avg_disp = displacement[moved_mask].mean().item()
                        events.append((
                            f"Object moved: '{obj_name}' displaced {avg_disp:.3f}m",
                            StatusCode.OBJECT_MOVED,
                            moved_mask.clone()
                        ))
                        if verbose:
                            envs = moved_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
                            print(f"[EventTracker] envs {envs}: Object moved: '{obj_name}'")

                    bumped_mask = stopped_with_start & (displacement >= self.bump_threshold) & (displacement < self.move_threshold)
                    if bumped_mask.any():
                        avg_disp = displacement[bumped_mask].mean().item()
                        # P52: nudging an object the task is about is the policy doing
                        # its job — a neutral note, not a red flag.
                        is_target = ~self._get_not_intended_mask(obj_name, per_env_intended)
                        for code, mask_sel, label in (
                            (StatusCode.TARGET_OBJECT_BUMPED, bumped_mask & is_target, "Target object bumped"),
                            (StatusCode.OBJECT_BUMPED, bumped_mask & ~is_target, "Object bumped"),
                        ):
                            if mask_sel.any():
                                events.append((f"{label}: '{obj_name}' nudged {avg_disp:.3f}m", code, mask_sel.clone()))
                        if verbose:
                            envs = bumped_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
                            print(f"[EventTracker] envs {envs}: Object bumped: '{obj_name}'")

                    # Clear start positions for stopped envs
                    self._started_moving_mask[obj_name] &= ~stopped_with_start

                self._object_is_moving[obj_name] = is_moving

            except Exception:
                continue

        return events

    def _check_out_of_scene_batched(
        self, env, per_env_intended, ignore_set, active_mask, verbose
    ) -> list[tuple[str, StatusCode, torch.Tensor]]:
        events = []
        world = get_world(env)

        for obj_name in world.objects.keys():
            if obj_name in ignore_set:
                continue

            not_intended = self._get_not_intended_mask(obj_name, per_env_intended)
            already_recorded = self._recorded_out_of_scene.get(
                obj_name, torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            )
            eligible = not_intended & active_mask & ~already_recorded

            if not eligible.any():
                continue

            try:
                current_pos, _ = world.get_pose(obj_name, env_id=None)  # (N, 3)
                outside = self._is_outside_workspace_batched(current_pos)
                new_outside = outside & eligible

                if new_outside.any():
                    events.append((
                        f"Object out of scene: '{obj_name}'",
                        StatusCode.OBJECT_OUT_OF_SCENE,
                        new_outside.clone()
                    ))
                    if obj_name not in self._recorded_out_of_scene:
                        self._recorded_out_of_scene[obj_name] = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                    self._recorded_out_of_scene[obj_name] |= new_outside
                    if verbose:
                        envs = new_outside.nonzero(as_tuple=False).squeeze(-1).tolist()
                        print(f"[EventTracker] envs {envs}: Object out of scene: '{obj_name}'")

            except Exception:
                continue

        return events

    def _check_tipped_objects_batched(
        self, env, upright_objects, active_mask, verbose
    ) -> list[tuple[str, StatusCode, torch.Tensor]]:
        events = []

        for obj_name in upright_objects:
            already_recorded = self._recorded_tipped_objects.get(
                obj_name, torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            )
            eligible = active_mask & ~already_recorded
            if not eligible.any():
                continue

            try:
                # object_upright returns (N,) bool when env_id=None
                is_upright = object_upright(env, obj_name, tolerance=0.3, env_id=None)
                tipped = ~is_upright & eligible

                if tipped.any():
                    events.append((
                        f"Object tipped over: '{obj_name}'",
                        StatusCode.OBJECT_TIPPED_OVER,
                        tipped.clone()
                    ))
                    if obj_name not in self._recorded_tipped_objects:
                        self._recorded_tipped_objects[obj_name] = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                    self._recorded_tipped_objects[obj_name] |= tipped
                    if verbose:
                        envs = tipped.nonzero(as_tuple=False).squeeze(-1).tolist()
                        print(f"[EventTracker] envs {envs}: Object tipped over: '{obj_name}'")

            except Exception:
                continue

        return events

    def _check_target_dropped_batched(
        self, env, per_env_intended, active_mask, verbose
    ) -> list[tuple[str, StatusCode, torch.Tensor]]:
        events = []

        # Check if any target is currently grabbed per env
        any_grabbed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Collect all unique intended objects across envs
        all_intended = set()
        for s in per_env_intended:
            all_intended.update(s)

        for obj_name in all_intended:
            try:
                grabbed = object_grabbed(env, obj_name, env_id=None)  # (N,) bool
                # Only count for envs where this object IS intended
                is_intended = torch.tensor(
                    [obj_name in per_env_intended[eid] for eid in range(self.num_envs)],
                    dtype=torch.bool, device=self.device
                )
                any_grabbed |= (grabbed & is_intended)
            except Exception:
                continue

        # TARGET_OBJECT_DROPPED is superseded by the GraspTracker's OBJECT_RELEASED /
        # OBJECT_DROPPED (findings.md B6): the state is still maintained for
        # re-grab bookkeeping, but nothing is emitted here any more.
        dropped = self._target_was_grabbed & ~any_grabbed & active_mask & ~self._recorded_target_dropped
        if dropped.any():
            self._recorded_target_dropped |= dropped

        self._target_was_grabbed = any_grabbed

        # Reset drop recording for envs that grab again
        re_grabbed = any_grabbed & self._recorded_target_dropped
        self._recorded_target_dropped &= ~re_grabbed

        return events

    def _check_gripper_object_collision_batched(
        self, env, per_env_intended, ignore_set, active_mask, verbose
    ) -> list[tuple[str, StatusCode, torch.Tensor]]:
        events = []
        world = get_world(env)

        candidates = [
            obj for obj in world.objects.keys()
            if obj not in ignore_set
        ]

        for obj_name in candidates:
            not_intended = self._get_not_intended_mask(obj_name, per_env_intended)
            already_recorded = self._recorded_gripper_hit_objects.get(
                obj_name, torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            )
            eligible = not_intended & active_mask & ~already_recorded

            if not eligible.any():
                continue

            try:
                contact = in_contact(world, "gripper", obj_name, env_id=None)  # (N,) bool
                new_contact = contact & eligible

                if new_contact.any():
                    events.append((
                        f"Gripper hit object: '{obj_name}'",
                        StatusCode.GRIPPER_HIT_OBJECT,
                        new_contact.clone()
                    ))
                    if obj_name not in self._recorded_gripper_hit_objects:
                        self._recorded_gripper_hit_objects[obj_name] = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                    self._recorded_gripper_hit_objects[obj_name] |= new_contact
                    if verbose:
                        envs = new_contact.nonzero(as_tuple=False).squeeze(-1).tolist()
                        print(f"[EventTracker] envs {envs}: Gripper hit object: '{obj_name}'")

            except Exception:
                continue

        return events

    def _check_multiple_objects_grabbed_batched(
        self, env, ignore_set, active_mask, verbose
    ) -> list[tuple[str, StatusCode, torch.Tensor]]:
        events = []
        eligible = active_mask & ~self._recorded_multiple_grab

        if not eligible.any():
            return events

        world = get_world(env)

        # Count per concrete gripper, not the "gripper" alias group: on a
        # bimanual robot, one hand clutching several objects is a violation,
        # but two hands holding one object each is normal behavior.
        multi = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for gripper in world.resolve_contact_bodies("gripper"):
            contact_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            for obj_name in world.objects.keys():
                if obj_name in ignore_set:
                    continue
                try:
                    contact = in_contact(world, gripper, obj_name, env_id=None)  # (N,) bool
                    contact_count += contact.int()
                except Exception:
                    continue
            multi |= contact_count > 1

        multi = multi & eligible
        if multi.any():
            events.append((
                "Multiple objects grabbed",
                StatusCode.MULTIPLE_OBJECTS_GRABBED,
                multi.clone()
            ))
            self._recorded_multiple_grab |= multi
            if verbose:
                envs = multi.nonzero(as_tuple=False).squeeze(-1).tolist()
                print(f"[EventTracker] envs {envs}: Multiple objects grabbed")

        return events
