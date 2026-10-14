"""Contact adapters without Kit/GPU; real-scene validation is separate."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("contacts_under_test", Path(__file__).resolve().parents[2]
    / "robolab/core/environments/newton_contacts.py")
contacts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contacts)

PAD = "/World/robot/Gripper/Robotiq_2F_85/left_inner_finger/Defeatured_2F_85_PAD_OPEN_fingertipsstep_01/pad"


class Array:
    def __init__(self, value):
        self.value = np.asarray(value).copy()
        self.shape = self.value.shape

    def numpy(self):
        return self.value

    def assign(self, value):
        self.value[:] = value

    def fill_(self, value):
        self.value[:] = value


def test_only_pads_are_simplified_and_restored_on_error():
    calls = []
    class Builder:
        shape_label = [PAD, "/scene/bottle", PAD + "_visual"]
        def approximate_meshes(self, **kw):
            calls.append(kw)
            return set(kw["shape_indices"])
    original = Builder.approximate_meshes
    reports = []
    with pytest.raises(RuntimeError, match="test exit"):
        with contacts.single_hull_fingertips(Builder, reports):
            assert Builder().approximate_meshes(method="coacd", shape_indices=[0, 1, 2],
                keep_visual_shapes=True, threshold=.1) == {0, 1, 2}
            raise RuntimeError("test exit")
    assert Builder.approximate_meshes is original
    assert calls[0] == dict(method="convex_hull", shape_indices=[0], keep_visual_shapes=True, raise_on_failure=True)
    assert calls[1]["shape_indices"] == [1, 2]
    assert calls[1]["threshold"] == .1
    assert len(reports) == 1


def test_capacity_guard_rejects_overflow_and_restores():
    calls = []
    original = lambda *a, **kw: calls.append(kw)
    wp = NS(launch=original)
    kernel = NS(key="ccd_kernel", adj=NS(args=[NS(label="ncollision_in"), NS(label="collision_pair_in")]))
    count = Array([4])
    pairs = Array(np.zeros((4, 2)))
    with contacts.guard_ccd_capacity(wp, {}):
        wp.launch(kernel, inputs=[count, pairs])
        count.value[0] = 5
        with pytest.raises(RuntimeError, match="overflow"):
            wp.launch(kernel, inputs=[count, pairs])
    assert len(calls) == 1
    assert wp.launch is original


def test_contact_arrays_match_and_other_impedance_fields_survive():
    impedance = np.tile([.9, .95, .001, .5, 2], (3, 1))
    solver = NS(mj_model=NS(geom_solimp=impedance.copy(), geom_solref=np.zeros((3, 2))),
        mjw_model=NS(geom_solimp=Array(impedance[None]), geom_solref=Array(np.zeros((1, 3, 2))),
                     opt=NS(timestep=Array([1 / 480]))))
    contacts.override_contact_response(solver, .0045, NS(vec2=lambda *x: x))
    contacts.override_contact_impedance(solver, (.99, .999))
    np.testing.assert_array_equal(solver.mjw_model.geom_solref.numpy()[0], solver.mj_model.geom_solref)
    np.testing.assert_array_equal(solver.mjw_model.geom_solimp.numpy()[0], solver.mj_model.geom_solimp)
    np.testing.assert_array_equal(solver.mj_model.geom_solimp[:, 2:], impedance[:, 2:])
    with pytest.raises(ValueError, match="twice"):
        contacts.override_contact_response(solver, .001, NS())
    with pytest.raises(ValueError, match="Impedance"):
        contacts.override_contact_impedance(solver, (.999, .99))


def test_unknown_kernel_layout_fails_closed():
    wp = NS(launch=lambda *a, **kw: pytest.fail("Unsafe kernel launched"))
    with contacts.guard_ccd_capacity(wp, {}):
        with pytest.raises(RuntimeError, match="Unsupported CCD"):
            wp.launch(NS(key="ccd_kernel", adj=NS(args=[])), inputs=[])


@pytest.mark.parametrize("filter_enabled", [False, True])
@pytest.mark.parametrize("rolling", [None, .001])
def test_runtime_patches_once_and_restores_on_failure(monkeypatch, filter_enabled, rolling):
    import sys
    updates = []
    class Solver:
        def _mujoco_warp_step(self):
            return 42
    class Builder:
        def approximate_meshes(self):
            pass
    original = Solver._mujoco_warp_step
    launch = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "warp", NS(launch=launch))
    monkeypatch.setitem(sys.modules, "newton", NS(ModelBuilder=Builder, BodyFlags=NS(KINEMATIC=2)))
    monkeypatch.setitem(sys.modules, "newton.solvers", NS(SolverMuJoCo=Solver))
    monkeypatch.setattr(contacts, "override_contact_response", lambda *a: updates.append("response"))
    monkeypatch.setattr(contacts, "override_contact_impedance", lambda *a: updates.append("impedance"))
    monkeypatch.setitem(sys.modules, "robolab.core.environments.newton_collision_filter",
                        NS(filter_immovable_contacts=lambda *a: updates.append("filter"),
                           override_support_rolling=lambda *a: updates.append("rolling")))
    from contextlib import nullcontext
    monkeypatch.setitem(sys.modules, "robolab.core.environments.newton_candidates",
                        NS(candidate_storage=lambda wp: nullcontext()))
    settings = dict(single_hull_fingertips=False, timeconst=.0045, impedance=(.99, .999),
                    filter_immovable_contacts=filter_enabled, support_rolling_friction=rolling)
    with pytest.raises(ValueError, match="exit"):
        with contacts.contact_runtime(settings):
            solver = Solver()
            assert solver._mujoco_warp_step() == 42
            assert solver._mujoco_warp_step() == 42
            assert updates == ((["filter"] if filter_enabled else [])
                               + (["rolling"] if rolling is not None else [])
                               + ["response", "impedance"])
            with pytest.raises(RuntimeError, match="nested"):
                with contacts.contact_runtime(settings):
                    pass
            raise ValueError("exit")
    assert Solver._mujoco_warp_step is original
    assert sys.modules["warp"].launch is launch
    assert contacts._active is False


def test_time_constant_boundary_accepts_only_float32_roundoff():
    dt = np.float32(1 / 240)
    solver = NS(mj_model=NS(geom_solref=np.zeros((1, 2))),
                mjw_model=NS(geom_solref=Array(np.zeros((1, 1, 2))), opt=NS(timestep=Array([dt]))))
    contacts.override_contact_response(solver, 2 / 240, NS(vec2=lambda *x: x))
    with pytest.raises(ValueError, match="twice"):
        contacts.override_contact_response(solver, (2 / 240) * .999, NS())
