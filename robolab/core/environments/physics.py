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
        raise RuntimeError("Newton requires Isaac Lab 3 / Isaac Sim 6; install the isaac60 extra.")
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
        num_substeps=1,
        use_cuda_graph=True,
    )


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


def physics_context(cfg, backend="physx", launcher_args=None):
    """Register the Newton runtime for the entire environment lifetime."""
    if backend == "physx":
        return nullcontext()
    from argparse import Namespace
    from isaaclab_tasks.utils.sim_launcher import launch_simulation

    return launch_simulation(cfg, launcher_args or Namespace(headless=True, enable_cameras=True, device=cfg.sim.device))


@contextmanager
def managed_env(scene, *, physics="physx", launcher_args=None, **kwargs):
    """Create and close a RoboLab environment within its backend runtime.

    Call after AppLauncher, as with create_env. Keep this context open while
    stepping and exporting episodes. The existing application owns Kit shutdown.
    """
    from robolab.core.environments.config import parse_env_cfg
    from robolab.core.environments.runtime import create_env

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
    cfg.physics_backend = physics
    with physics_context(cfg, physics, launcher_args):
        env, cfg = create_env(cfg, **kwargs)
        try:
            yield env, cfg
        finally:
            env.close()
