# SPDX-License-Identifier: Apache-2.0
"""Physics resolution controls that preserve the authored control/render periods."""
from math import isclose, isfinite


def configure_timing(cfg, backend, sim_rate=None, newton_substeps=None):
    if newton_substeps is not None:
        if backend != "newton_mjwarp":
            raise ValueError("--newton-substeps requires --physics newton_mjwarp")
        if isinstance(newton_substeps, bool) or not isinstance(newton_substeps, int) or newton_substeps < 1:
            raise ValueError("Newton substeps must be a positive integer")
        cfg.newton_substeps = newton_substeps
    if sim_rate is None:
        return
    if isinstance(sim_rate, bool) or not isfinite(sim_rate) or sim_rate <= 0:
        raise ValueError("Simulation rate must be finite and positive")
    old_dt = cfg.sim.dt
    periods = {"decimation": old_dt * cfg.decimation,
               "render_interval": old_dt * cfg.sim.render_interval}
    counts = {}
    for name, period in periods.items():
        count = period * sim_rate
        if count < 1 or not isclose(count, round(count), rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"--sim-rate {sim_rate:g} cannot preserve the authored {name} period "
                             f"({period:g} s); choose an integer multiple of {1 / period:g} Hz")
        counts[name] = round(count)
    cfg.sim.dt = 1 / sim_rate
    cfg.decimation = counts["decimation"]
    cfg.sim.render_interval = counts["render_interval"]


def timing_provenance(cfg, backend, sim_rate=None, newton_substeps=None):
    substeps = cfg.sim.physics.num_substeps if backend == "newton_mjwarp" else 1
    return {"requested_sim_rate_hz": sim_rate, "requested_newton_substeps": newton_substeps,
            "sim_rate_hz": 1 / cfg.sim.dt, "physics_dt_s": cfg.sim.dt,
            "control_rate_hz": 1 / (cfg.sim.dt * cfg.decimation), "decimation": cfg.decimation,
            "render_interval": cfg.sim.render_interval,
            "render_rate_hz": 1 / (cfg.sim.dt * cfg.sim.render_interval),
            "newton_substeps": substeps if backend == "newton_mjwarp" else None,
            "solver_dt_s": cfg.sim.dt / substeps}
