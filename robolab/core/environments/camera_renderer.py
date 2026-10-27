"""Camera renderer selection: Isaac RTX through Kit, or Kit-less OVRTX.

Isaac RTX keeps each camera's existing configuration. OVRTX replaces every
camera's renderer before the environment is built; the process must then not
start Kit (no AppLauncher), since OVRTX cannot share a process with it.
"""

RENDERERS = ("isaacrtx", "ovrtx")


def configure_camera_renderer(cfg, renderer, *, log_dir=None):
    """Select the camera renderer on a parsed env cfg, before create_env."""
    if renderer not in RENDERERS:
        raise ValueError(f"Unknown camera renderer '{renderer}'; expected one of {list(RENDERERS)}.")
    from isaaclab.sensors import CameraCfg

    cfg.camera_renderer = renderer
    cfg.experimental_kitless = renderer == "ovrtx"
    if renderer == "isaacrtx":
        return cfg
    from pathlib import Path

    from isaaclab_ov.renderers import OVRTXRendererCfg

    # The settings validated against Isaac RTX on 2026-10-24 (docs/verified/ovrtx-*).
    options = dict(async_rendering=False, enable_shadows=True, log_level="warn")
    if log_dir is not None:
        options["log_file_path"] = str(Path(log_dir, "ovrtx.log"))
    for camera in vars(cfg.scene).values():
        if not isinstance(camera, CameraCfg):
            continue
        # OVRTX keeps the spawn viewpoint of attached cameras (the wrist camera)
        # unless the pose is refreshed every update.
        camera.update_latest_camera_pose = True
        camera.renderer_cfg = OVRTXRendererCfg(**options)
    return cfg
