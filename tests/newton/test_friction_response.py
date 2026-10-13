# SPDX-License-Identifier: Apache-2.0
"""A material override must alter actual native-contact dynamics."""
import numpy as np
import pytest


def test_sliding_distance_responds_to_requested_friction(tmp_path):
    newton = pytest.importorskip("newton")
    wp = pytest.importorskip("warp")
    from newton.solvers import SolverMuJoCo
    from robolab.core.physics.newton_friction import apply, read_solver, check_applied
    wp.config.kernel_cache_dir = str(tmp_path / "warp-cache")
    distances = []
    for coefficient in (.1, .9):
        builder = newton.ModelBuilder()
        body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, .0501), wp.quat_identity()), label="/box")
        builder.add_shape_box(body, hx=.05, hy=.05, hz=.05, label="/box/collider")
        builder.add_ground_plane(label="/floor")
        model = builder.finalize(device="cpu")
        solver = SolverMuJoCo(model, use_mujoco_cpu=False, use_mujoco_contacts=True,
                              solver="newton", nconmax=128, njmax=256)
        targets = {"objects": {"box": "/box", "floor": "/floor"}, "gripper": {}}
        requested = {"mode": "uniform", "gripper_bodies": [], "objects": {
            name: {"static": coefficient, "dynamic": coefficient} for name in ("box", "floor")}}
        apply(solver, requested, targets)
        assert not check_applied(read_solver(solver, requested, targets))
        state, out, control = model.state(), model.state(), model.control()
        velocity = state.joint_qd.numpy()
        velocity[0] = .5
        state.joint_qd.assign(velocity)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        for _ in range(240):
            solver.step(state, out, control, None, 1 / 240)
            state, out = out, state
        pose = state.body_q.numpy()[0]
        assert np.isfinite(pose).all() and abs(pose[2] - .05) < .001
        distances.append(float(pose[0]))
    assert .08 < distances[0] < .18
    assert .005 < distances[1] < .03
    assert distances[0] > 3 * distances[1]
