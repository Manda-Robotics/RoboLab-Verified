"""Version-independent application of RoboLab's RTX renderer selection."""

_RTX_MODES = {"realtime": "RaytracedLighting", "pathtracing": "PathTracing"}


def apply_render_settings(env_cfg, renderer, rendering_mode):
    if renderer not in _RTX_MODES:
        raise ValueError(f"Unknown renderer '{renderer}'; expected one of {list(_RTX_MODES)}.")
    legacy = getattr(env_cfg.sim, "render", None)
    if legacy is not None:
        if rendering_mode is not None:
            legacy.rendering_mode = rendering_mode
        targets = [legacy]
    else:
        # Lab 3 source moved global RTX settings onto camera renderer configs.
        # The old performance/balanced/quality presets have no equivalent field;
        # fail explicitly rather than recording a preset we did not apply.
        if rendering_mode is not None:
            raise ValueError("Isaac Lab's 6.1 stack removed rendering quality presets; omit --rendering-type "
                             "and configure camera.renderer_cfg.global_settings explicitly.")
        targets = []
        for sensor in vars(env_cfg.scene).values():
            cfg = getattr(sensor, "renderer_cfg", None)
            if getattr(cfg, "renderer_type", None) == "isaac_rtx":
                targets.append(cfg.global_settings)
    for target in targets:
        settings = dict(target.carb_settings or {})
        settings["/rtx/rendermode"] = _RTX_MODES[renderer]
        target.carb_settings = settings
