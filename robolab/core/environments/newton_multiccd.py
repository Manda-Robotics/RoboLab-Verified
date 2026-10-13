# SPDX-License-Identifier: Apache-2.0
"""Enable native multi-point contacts on the pinned GPU solver."""
from contextlib import contextmanager
from functools import wraps
import inspect


@contextmanager
def native_multiccd(solver_class):
    """Bridge the constructor option missing from the pinned Isaac Lab config.

    Newton 1.5.2 defaults to a single contact for convex mesh pairs. Flat
    supports then rock under load. Let Newton enable MULTICCD and apply its
    native-CCD zero-margin handling before compilation and GPU upload. CPU
    solvers and Newton-generated contacts retain their original options.
    Scope this adapter to the owning simulation, alongside contact storage.
    """
    original = solver_class.__init__
    signature = inspect.signature(original)

    @wraps(original)
    def initialize(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        if not bound.arguments['use_mujoco_cpu'] and bound.arguments['use_mujoco_contacts']:
            kwargs['enable_multiccd'] = True
        return original(self, *args, **kwargs)

    solver_class.__init__ = initialize
    try:
        yield
    finally:
        solver_class.__init__ = original
