# SPDX-License-Identifier: Apache-2.0
"""Avoid a redundant CPU contact solve while constructing a GPU-only solver."""
from contextlib import contextmanager
from functools import wraps


@contextmanager
def defer_initial_cpu_contacts(solver_class, mujoco):
    """Keep Newton's initialization FK/equalities, defer contacts to MJWarp.

    Pinned Newton 1.5.2 calls mj_forward on a CPU staging MjData before uploading
    the model/data to MJWarp. Dense scenes can overflow MuJoCo's small automatic
    CPU arena there, although the configured GPU contact/constraint budgets are
    sufficient. The initial CPU contact solve is not used to advance physics.

    Disable only CONTACT, only for that solver's staging forward call. Restore
    the model options before GPU upload, and never alter CPU solvers or unrelated
    forward calls. The ordinary guarded GPU step generates all physical contacts.
    The caller owns the pinned runtime's single-simulation lifetime.
    """
    original_convert = solver_class._convert_to_mjc

    @wraps(original_convert)
    def convert(self, *args, **kwargs):
        if self.use_mujoco_cpu or not self._use_mujoco_contacts:
            return original_convert(self, *args, **kwargs)
        original_forward = mujoco.mj_forward

        def forward(model, data):
            if model is not self.mj_model or data is not self.mj_data:
                return original_forward(model, data)
            flags = model.opt.disableflags
            try:
                model.opt.disableflags = flags | int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
                return original_forward(model, data)
            finally:
                model.opt.disableflags = flags

        mujoco.mj_forward = forward
        try:
            return original_convert(self, *args, **kwargs)
        finally:
            mujoco.mj_forward = original_forward

    solver_class._convert_to_mjc = convert
    try:
        yield
    finally:
        solver_class._convert_to_mjc = original_convert
