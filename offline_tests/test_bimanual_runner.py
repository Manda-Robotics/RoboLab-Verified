"""The scripted bimanual client — the thing that makes the two-arm rigs runnable.

No released checkpoint drives two arms, so without this client the rigs could not be
started at all and every claim about them was offline-only. These tests pin the two
properties that make it safe to point at a real robot: the action is laid out for the
rig it is actually driving, and joint-position commands are built from the arm's
current pose rather than from zero.
"""
import numpy as np
import pytest
import torch

from policies.bimanual.client import JOINTPOS_DIM, RELIK_DIM, ScriptedBimanualClient


def obs(n_arm: int, n_finger: int, *, left=None, right=None, finger=0.03, num_envs=1):
    """A proprio observation shaped like the rig's, batched over envs."""
    left = np.arange(n_arm, dtype=np.float32) * 0.1 if left is None else np.asarray(left, np.float32)
    right = -left if right is None else np.asarray(right, np.float32)
    rep = lambda v: torch.tensor(np.tile(np.asarray(v, np.float32), (num_envs, 1)))
    return {"proprio_obs": {
        "left_arm_joint_pos": rep(left),
        "right_arm_joint_pos": rep(right),
        "left_gripper_pos": rep(np.full(n_finger, finger, np.float32)),
        "right_gripper_pos": rep(np.full(n_finger, finger, np.float32)),
    }}


FRANKA = dict(n_arm=7, n_finger=1)      # [7, 1, 7, 1]
ALOHA = dict(n_arm=6, n_finger=2)       # [6, 2, 6, 2]


@pytest.mark.parametrize("rig", [FRANKA, ALOHA], ids=["dual_franka", "aloha"])
def test_the_action_is_the_size_the_rig_expects(rig):
    c = ScriptedBimanualClient()
    a = c.infer(obs(**rig), "", env_id=0)["action"]
    assert a.shape == (JOINTPOS_DIM,) and a.dtype == np.float32


@pytest.mark.parametrize("rig", [FRANKA, ALOHA], ids=["dual_franka", "aloha"])
def test_arm_joints_land_in_arm_slots_not_finger_slots(rig):
    """The failure this guards against is silent: an arm joint written into a finger
    slot bends the arm and nothing errors."""
    n, f = rig["n_arm"], rig["n_finger"]
    left = np.arange(1, n + 1, dtype=np.float32)
    right = np.arange(101, 101 + n, dtype=np.float32)
    c = ScriptedBimanualClient(amplitude_rad=0.0)          # no wobble, so it is exact
    a = c.infer(obs(**rig, left=left, right=right), "", env_id=0)["action"]
    assert np.allclose(a[0:n], left)
    assert np.allclose(a[n + f:2 * n + f], right)


def test_joint_targets_are_absolute_and_track_the_current_pose():
    """A zero action would command every joint to 0 rad and slam the arms down."""
    c = ScriptedBimanualClient(amplitude_rad=0.0)
    pose = np.array([0.4, -0.7, 0.2, -2.1, 0.0, 1.5, 0.8], np.float32)
    a = c.infer(obs(**FRANKA, left=pose, right=pose), "", env_id=0)["action"]
    assert np.allclose(a[0:7], pose) and np.allclose(a[8:15], pose)
    assert np.abs(a).sum() > 1.0, "the action is not just zeros"


def test_the_wobble_moves_an_elbow_and_leaves_the_wrist_alone():
    c = ScriptedBimanualClient(amplitude_rad=0.2, period_s=4.0, control_hz=15.0)
    o = obs(**FRANKA, left=np.zeros(7, np.float32), right=np.zeros(7, np.float32))
    moved = set()
    for _ in range(30):
        a = c.infer(o, "", env_id=0)["action"]
        moved |= {i for i in range(7) if abs(a[i]) > 1e-6}
    assert moved == {3}, moved


def test_aloha_fingers_are_held_in_metres_not_commanded_zero_to_one():
    """ALOHA finger joints are a distance. A 0/1 gripper command is a metre of travel."""
    c = ScriptedBimanualClient(amplitude_rad=0.0)
    a = c.infer(obs(**ALOHA, finger=0.024), "", env_id=0)["action"]
    assert np.allclose(a[6:8], 0.024) and np.allclose(a[14:16], 0.024)


def test_the_franka_gripper_is_a_binary_channel_that_both_opens_and_closes():
    c = ScriptedBimanualClient(period_s=2.0, control_hz=15.0)
    o = obs(**FRANKA)
    seen = {float(c.infer(o, "", env_id=0)["action"][7]) for _ in range(40)}
    assert seen == {0.0, 1.0}, seen


def test_rel_ik_is_fourteen_deltas_centred_on_zero():
    c = ScriptedBimanualClient(action_space="rel_ik")
    a = c.infer(obs(**FRANKA), "", env_id=0)["action"]
    assert a.shape == (RELIK_DIM,)
    assert abs(a[0]) < 1e-9 and abs(a[1]) < 1e-9      # no lateral drift


def test_the_motion_is_deterministic_so_a_smoke_test_is_reproducible():
    o = obs(**FRANKA)
    runs = []
    for _ in range(2):
        c = ScriptedBimanualClient()
        c.begin_episode(0)
        runs.append(np.stack([c.infer(o, "", env_id=0)["action"] for _ in range(20)]))
    assert np.array_equal(runs[0], runs[1])


def test_each_env_gets_its_own_phase_counter():
    """Env 1 must not inherit env 0's phase: the arms would jump on the first step."""
    c = ScriptedBimanualClient()
    o = obs(**FRANKA, num_envs=2)
    for _ in range(5):
        c.infer(o, "", env_id=0)
    assert np.array_equal(c.infer(o, "", env_id=1)["action"],
                          ScriptedBimanualClient().infer(o, "", env_id=1)["action"])


def test_a_rig_whose_arms_do_not_fit_sixteen_numbers_is_refused():
    c = ScriptedBimanualClient()
    with pytest.raises(ValueError, match="cannot lay out"):
        c.infer(obs(n_arm=8, n_finger=1), "", env_id=0)


def test_an_unknown_action_space_is_refused_at_construction():
    with pytest.raises(ValueError, match="action_space"):
        ScriptedBimanualClient(action_space="cartesian")
