"""Optional physics backends. Imports stay lazy for Isaac Sim 5 compatibility."""

from contextlib import contextmanager, nullcontext
from pathlib import Path


def configure_physics(cfg, backend="physx"):
    """Select physics before constructing the simulation; retain authored timing."""
    if backend == "physx":
        return
    if backend != "newton_mjwarp":
        raise ValueError(f"Unknown physics backend: {backend}")
    if not hasattr(cfg.sim, "physics"):
        raise RuntimeError("Newton requires the Isaac Lab 3 source stack; run scripts/install_newton61.py.")
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

    from robolab.constants import ROBOTS_DIR

    robot = cfg.scene.robot
    if Path(robot.spawn.usd_path).name != "franka_robotiq_2f_85_isaac60.usd":
        raise ValueError("The experimental Newton preset currently supports the DROID robot only.")
    robot.spawn.usd_path = str(Path(ROBOTS_DIR) / "franka_robotiq_2f_85_newton.usda")
    if not Path(robot.spawn.usd_path).is_file():
        raise FileNotFoundError(f"Newton robot overlay missing: {robot.spawn.usd_path}")
    cfg.sim.physics = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=4096, nconmax=2048, iterations=100,
            ls_iterations=50, integrator="implicitfast",
        ),
        num_substeps=4,
        use_cuda_graph=True,
    )
    if getattr(cfg, "newton_gripper_stabilization", True):
        from math import pi
        from robolab.robots.newton_gripper import RateLimitedGripperActionCfg

        gripper = robot.actuators["gripper"]
        gripper.stiffness, gripper.damping = 10.0, 2.0
        # Keep the policy's existing binary semantics. Only the target sent to
        # physics is rate limited; this is recorded in the exported action cfg.
        cfg.actions.finger_joint = RateLimitedGripperActionCfg(
            asset_name="robot", joint_names=["finger_joint"],
            open_command_expr={"finger_joint": 0.0}, close_command_expr={"finger_joint": pi / 4})


def ensure_output_backend(output_dir, backend):
    """Do not resume another solver's episodes into this experiment."""
    import json

    directory = Path(output_dir)
    marker = directory / "physics_backend.json"
    if marker.exists():
        previous = json.loads(marker.read_text())["physics_backend"]
        if previous != backend:
            raise ValueError(f"Output contains {previous} results; choose a new output folder for {backend}.")
    elif backend != "physx" and directory.exists() and any(directory.iterdir()):
        raise ValueError("Newton requires a fresh output folder or an existing Newton backend marker.")
    directory.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"physics_backend": backend}, indent=2) + "\n")


@contextmanager
def physics_context(cfg, backend="physx", launcher_args=None):
    """Register the Newton runtime for the entire environment lifetime."""
    disable_graph = getattr(launcher_args, "newton_disable_cuda_graph", False)
    if backend == "physx":
        if disable_graph:
            raise ValueError("Disabling the Newton CUDA graph requires --physics newton_mjwarp")
        yield
        return
    if disable_graph:
        cfg.sim.physics.use_cuda_graph = False
    from argparse import Namespace
    try:
        from isaaclab.app.sim_launcher import launch_simulation
        native_mesh_import = True
    except ModuleNotFoundError as exc:
        if exc.name not in {"isaaclab", "isaaclab.app", "isaaclab.app.sim_launcher"}:
            raise
        from isaaclab_tasks.utils.sim_launcher import launch_simulation
        native_mesh_import = False

    threshold = getattr(cfg, "newton_convex_decomposition_threshold", None)
    contact_timeconst = getattr(cfg, "newton_contact_timeconst", None)
    if native_mesh_import and (threshold is not None or contact_timeconst is not None):
        raise ValueError(
            "The Isaac Sim 6.1 stack uses native authored mesh import. Remove the beta2 "
            "decomposition/contact overrides; their Newton 1.2 shim is not compatible."
        )
    if contact_timeconst is not None:
        from math import isfinite
        if (not isfinite(contact_timeconst) or contact_timeconst < 2 * cfg.sim.dt / cfg.sim.physics.num_substeps):
            raise ValueError("Contact time constant must be finite and at least twice the solver timestep")
        if threshold is None:
            raise ValueError("Contact response diagnostic requires the authored mesh-import option")
    meshes = nullcontext()
    if threshold is not None:
        from robolab.core.environments.newton_meshes import mesh_decomposition_context
        meshes = mesh_decomposition_context(threshold, contact_timeconst=contact_timeconst)
    # Modern Lab treats args.physics as a request to REPLACE the complete solver
    # config. We already configured it (including substeps and contact limits).
    # Copy the namespace so the policy's original arguments remain unchanged.
    launch_args = Namespace(**vars(launcher_args)) if launcher_args is not None else Namespace(
        headless=True, enable_cameras=True, device=cfg.sim.device)
    launch_args.physics = None
    # RoboLab uses Kit/USD services and calibrated Isaac RTX cameras even when
    # physics is Newton; do not let automatic renderer selection switch to OVRTX.
    launch_args.require_kit = True
    with meshes:
        with launch_simulation(cfg, launch_args):
            yield


@contextmanager
def managed_env(scene, *, physics="physx", launcher_args=None, **kwargs):
    """Create and close a RoboLab environment within its backend runtime.

    Call after AppLauncher, as with create_env. Keep this context open while
    stepping and exporting episodes. The existing application owns Kit shutdown.
    """
    from robolab.core.environments.config import parse_env_cfg
    from robolab.core.environments.runtime import create_env

    mesh_threshold = getattr(launcher_args, "newton_convex_decomposition_threshold", None)
    contact_timeconst = getattr(launcher_args, "newton_contact_timeconst", None)
    if getattr(launcher_args, "newton_disable_cuda_graph", False) and physics != "newton_mjwarp":
        raise ValueError("Disabling the Newton CUDA graph requires --physics newton_mjwarp")
    if (mesh_threshold is not None or contact_timeconst is not None) and physics != "newton_mjwarp":
        raise ValueError("Mesh/contact override requires --physics newton_mjwarp")
    if physics == "physx":
        env, cfg = create_env(scene, **kwargs)
        cfg.physics_backend = physics
        try:
            yield env, cfg
        finally:
            env.close()
        return
    cfg = parse_env_cfg(scene, **{
        k: v for k, v in kwargs.items()
        if k in {"device", "seed", "num_envs", "env_spacing", "eye", "lookat", "use_fabric"}
    }) if isinstance(scene, str) else scene
    configure_physics(cfg, physics)
    if mesh_threshold is not None:
        cfg.newton_convex_decomposition_threshold = mesh_threshold
    if contact_timeconst is not None:
        cfg.newton_contact_timeconst = contact_timeconst
    cfg.physics_backend = physics
    with physics_context(cfg, physics, launcher_args):
        env, cfg = create_env(cfg, **kwargs)
        try:
            yield env, cfg
        finally:
            env.close()
