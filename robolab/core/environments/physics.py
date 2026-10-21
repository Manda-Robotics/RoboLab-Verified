"""Optional physics backends. Imports stay lazy for Isaac Sim 5 compatibility."""

from contextlib import contextmanager, nullcontext
from pathlib import Path


def configure_physics(cfg, backend="physx"):
    """Select physics before constructing the simulation; retain authored timing."""
    if backend == "physx":
        return
    if backend != "newton_mjwarp":
        raise ValueError(f"Unknown physics backend: {backend}")
    from robolab.core.physics.friction import require_supported_backend

    require_supported_backend(backend, (getattr(cfg, "friction", None) or {}).get("mode", "upstream"))
    substeps = getattr(cfg, "newton_substeps", 4)
    if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
        raise ValueError("Newton substeps must be a positive integer")
    if not hasattr(cfg.sim, "physics"):
        raise RuntimeError("Newton requires the Isaac Lab 3 source stack; run scripts/install_newton61.py.")
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

    from robolab.constants import ROBOTS_DIR

    robot = cfg.scene.robot
    if Path(robot.spawn.usd_path).name != "franka_robotiq_2f_85_isaac60.usd":
        raise ValueError(
            "The experimental Newton preset supports only the default Isaac 6 DROID asset. "
            "Unset ROBOLAB_ROBOT_USD; custom and solid-pad variants need separate Newton overlays."
        )
    disable_gravity = getattr(getattr(robot.spawn, "rigid_props", None), "disable_gravity", False)
    overlay = ("franka_robotiq_2f_85_newton_nogravity.usda" if disable_gravity
               else "franka_robotiq_2f_85_newton.usda")
    robot.spawn.usd_path = str(Path(ROBOTS_DIR) / overlay)
    if not Path(robot.spawn.usd_path).is_file():
        raise FileNotFoundError(f"Newton robot overlay missing: {robot.spawn.usd_path}")
    from robolab.core.physics.newton_bagels import configure_scene

    clean_bagels = configure_scene(cfg)
    settings = dict(cone="elliptic" if clean_bagels else "pyramidal", enable_multiccd=True, compact_keyboard_colliders=True, compact_shelf_colliders=True, compact_contact_storage=True, mesh_compatibility=True, single_hull_fingertips=True, filter_immovable_contacts=True, support_rolling_friction=0.001, timeconst=0.0045,
                    impedance=(0.99, 0.999), contact_capacity=65536,
                    constraint_capacity=131072, gripper_stiffness=10.0,
                    gripper_damping=0.5, gripper_target_velocity=3.141592653589793)
    overrides = getattr(cfg, "newton_contact_settings", {})
    if set(overrides) - set(settings):
        raise ValueError("Unknown Newton contact setting")
    settings.update(overrides)
    # Keep the established contact response unless the coarser solver timestep
    # requires a longer time constant. Explicit unsafe requests still fail.
    if "timeconst" not in overrides:
        settings["timeconst"] = max(settings["timeconst"], 2 * cfg.sim.dt / substeps)
    from math import isfinite
    if settings["cone"] not in ("pyramidal", "elliptic"):
        raise ValueError("Newton cone must be pyramidal or elliptic")
    if not isinstance(settings["enable_multiccd"], bool):
        raise ValueError("enable_multiccd must be boolean")
    if not isinstance(settings["compact_keyboard_colliders"], bool):
        raise ValueError("compact_keyboard_colliders must be boolean")
    if not isinstance(settings["compact_shelf_colliders"], bool):
        raise ValueError("compact_shelf_colliders must be boolean")
    if not isinstance(settings["compact_contact_storage"], bool):
        raise ValueError("compact_contact_storage must be boolean")
    if not isinstance(settings["mesh_compatibility"], bool):
        raise ValueError("mesh_compatibility must be boolean")
    if not isinstance(settings["filter_immovable_contacts"], bool):
        raise ValueError("filter_immovable_contacts must be boolean")
    rolling = settings["support_rolling_friction"]
    if rolling is not None and (isinstance(rolling, bool) or not isfinite(rolling) or rolling <= 0):
        raise ValueError("support_rolling_friction must be a positive finite length in metres")
    for key in ("contact_capacity", "constraint_capacity"):
        if isinstance(settings[key], bool) or not isinstance(settings[key], int) or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("timeconst", "gripper_stiffness", "gripper_target_velocity", "gripper_damping"):
        if not isfinite(settings[key]) or settings[key] < 0 or (key != "gripper_damping" and settings[key] == 0):
            raise ValueError(f"Invalid Newton {key}")
    if settings["timeconst"] < 2 * cfg.sim.dt / substeps:
        raise ValueError("Newton contact time constant must be at least twice the solver timestep")
    impedance = settings["impedance"]
    if len(impedance) != 2 or not all(isfinite(x) for x in impedance) or not 0 < impedance[0] <= impedance[1] < 1:
        raise ValueError("Newton impedance requires 0 < DMIN <= DMAX < 1")
    cfg.newton_contact_settings = settings
    cfg.sim.physics = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=settings["constraint_capacity"], nconmax=settings["contact_capacity"], iterations=100,
            ls_iterations=50, integrator="implicitfast", cone=settings["cone"],
        ),
        num_substeps=substeps,
        use_cuda_graph=False,  # The pre-kernel capacity guard requires eager execution.
    )
    if getattr(cfg, "newton_gripper_stabilization", True):
        from math import pi
        from robolab.robots.newton_gripper import RateLimitedGripperActionCfg

        gripper = robot.actuators["gripper"]
        gripper.stiffness = settings["gripper_stiffness"]
        gripper.damping = settings["gripper_damping"]
        # Keep the policy's existing binary semantics. Only the target sent to
        # physics is rate limited; this is recorded in the exported action cfg.
        cfg.actions.finger_joint = RateLimitedGripperActionCfg(
            asset_name="robot", joint_names=["finger_joint"],
            open_command_expr={"finger_joint": 0.0}, close_command_expr={"finger_joint": pi / 4},
            max_target_velocity=settings["gripper_target_velocity"])
    if getattr(cfg, "friction", None) is not None:
        from robolab.core.physics.newton_friction import configure as configure_friction
        configure_friction(cfg)
    if getattr(cfg, "contact_extra_bodies", None):
        from robolab.core.sensors.contact_sensor_utils import configure_newton_extra_contacts

        configure_newton_extra_contacts(cfg)


def ensure_output_backend(output_dir, backend, parameters=None):
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
    payload = {"physics_backend": backend}
    if parameters is not None:
        if marker.exists():
            saved = json.loads(marker.read_text()).get("parameters")
            if saved is not None and saved != parameters:
                raise ValueError("Output contains different physics parameters; choose a new output folder.")
            if saved is None and any(directory.iterdir()):
                defaults = (parameters.get("friction", {}).get("mode") == "upstream"
                            and parameters.get("sim_rate_hz") is None
                            and parameters.get("newton_substeps") is None)
                if not defaults:
                    raise ValueError("Existing output has no physics-parameter provenance; choose a fresh output folder.")
        payload["parameters"] = parameters
    elif marker.exists():
        payload = json.loads(marker.read_text())
    directory.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(payload, indent=2) + "\n")


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
    contacts = nullcontext()
    if native_mesh_import and hasattr(cfg, "newton_contact_settings"):
        if cfg.sim.physics.use_cuda_graph:
            raise ValueError("The guarded Newton contact preset requires CUDA graphs disabled")
        from robolab.core.environments.newton_contacts import contact_runtime
        contacts = contact_runtime(cfg.newton_contact_settings)
    compatibility = nullcontext()
    if native_mesh_import and getattr(cfg, "newton_contact_settings", {}).get("mesh_compatibility", False):
        from robolab.core.environments.newton_mesh_compat import mesh_compatibility
        compatibility = mesh_compatibility()
    storage = nullcontext()
    initialization = nullcontext()
    if native_mesh_import and getattr(cfg, "newton_contact_settings", {}).get("compact_contact_storage", False):
        from robolab.core.environments.newton_native_contacts import native_contact_storage
        storage = native_contact_storage()
        import mujoco
        from newton.solvers import SolverMuJoCo
        from robolab.core.environments.newton_initialization import defer_initial_cpu_contacts

        initialization = defer_initial_cpu_contacts(SolverMuJoCo, mujoco)
    manifold = nullcontext()
    if native_mesh_import and getattr(cfg, "newton_contact_settings", {}).get("enable_multiccd", False):
        from newton.solvers import SolverMuJoCo
        from robolab.core.environments.newton_multiccd import native_multiccd
        manifold = native_multiccd(SolverMuJoCo)
    shelf = nullcontext()
    if native_mesh_import and getattr(cfg, "newton_contact_settings", {}).get("compact_shelf_colliders", False):
        from robolab.core.environments.newton_shelf import shelf_colliders
        shelf = shelf_colliders()
    keyboard = nullcontext()
    if native_mesh_import and getattr(cfg, "newton_contact_settings", {}).get("compact_keyboard_colliders", False):
        from robolab.core.environments.newton_keyboard import keyboard_colliders
        keyboard = keyboard_colliders()
    friction = nullcontext()
    if getattr(cfg, "newton_friction_targets", None) is not None:
        from robolab.core.physics.newton_friction import friction_runtime
        friction = friction_runtime(cfg.friction, cfg.newton_friction_targets)
    with storage, manifold, meshes, shelf, keyboard, compatibility, contacts, initialization, friction:
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
    sim_rate = getattr(launcher_args, "sim_rate", None)
    substeps = getattr(launcher_args, "newton_substeps", None)
    if substeps is not None and physics != "newton_mjwarp":
        raise ValueError("--newton-substeps requires --physics newton_mjwarp")
    if physics == "physx" and sim_rate is None:
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
    from robolab.core.physics.timing import configure_timing, timing_provenance
    configure_timing(cfg, physics, sim_rate, substeps)
    configure_physics(cfg, physics)
    cfg.physics_timing = timing_provenance(cfg, physics, sim_rate, substeps)
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
