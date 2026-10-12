# SPDX-License-Identifier: Apache-2.0
"""Avoid redundant Newton collision storage for native MuJoCo contacts only."""
from contextlib import contextmanager
from functools import wraps
import importlib.metadata
import inspect

import numpy as np


class _FilterView:
    """Supply implicit same-body exclusions lazily to the pinned mask compiler."""
    def __init__(self, model, selected):
        self._model = model
        self._selected = selected
        self._body = model.shape_body.numpy()

    def __getattr__(self, name):
        return getattr(self._model, name)

    def shape_collision_filter_mask(self, pairs):
        excluded = self._model.shape_collision_filter_mask(pairs)
        bodies = self._body[pairs]
        return excluded | ((bodies[:, 0] >= 0) & (bodies[:, 0] == bodies[:, 1]))

    def shape_collision_filter_pairs_array(self):
        # Only called by the exact compiler for at most 256 selected shapes.
        if len(self._selected) > 256:
            raise RuntimeError("Refusing to materialize a large same-body pair table")
        a, b = np.triu_indices(len(self._selected), k=1)
        pairs = np.stack((self._selected[a], self._selected[b]), axis=1)
        bodies = self._body[pairs]
        same = (bodies[:, 0] >= 0) & (bodies[:, 0] == bodies[:, 1])
        return np.concatenate((self._model.shape_collision_filter_pairs_array(), pairs[same]))


_active = False
_FLAG = "_robolab_native_contacts_only"


@contextmanager
def native_contact_storage():
    """Preserve physical filters without building same-body/unused pair lists.

    Newton 1.5.2 eagerly records every pair within each body during add_shape,
    then enumerates all candidate pairs during finalize. MuJoCo excludes same-
    body collisions itself and builds its own candidates. Retain explicit and
    cross-body exclusions. Supply implicit exclusions lazily to both collision
    mask compiler paths, so omitting their storage does not change graph
    coloring. Reject use of these models by Newton's collision pipeline.
    """
    global _active
    if _active:
        raise RuntimeError("Concurrent/nested native contact storage contexts are unsupported")
    if importlib.metadata.version("newton") != "1.5.2":
        raise RuntimeError("Re-audit native contact storage before changing Newton 1.5.2")
    from newton import CollisionPipeline, ModelBuilder
    from newton.solvers import SolverMuJoCo

    add_original = ModelBuilder.add_shape
    find_original = ModelBuilder._find_shape_contact_pairs
    solver_original = SolverMuJoCo.__init__
    pipeline_original = CollisionPipeline.__init__
    signature = inspect.signature(solver_original)
    masks_original = SolverMuJoCo.__dict__["_compile_newton_collision_masks"]
    colors_original = SolverMuJoCo.__dict__["_color_collision_shapes"]

    def compile_masks(model, selected_shapes):
        view = _FilterView(model, selected_shapes) if getattr(model, _FLAG, False) else model
        return masks_original.__func__(view, selected_shapes)

    def color_shapes(model, selected_shapes, *args, **kwargs):
        view = _FilterView(model, selected_shapes) if getattr(model, _FLAG, False) else model
        return colors_original.__func__(view, selected_shapes, *args, **kwargs)

    @wraps(add_original)
    def add_shape(builder, *args, **kwargs):
        body = kwargs.get("body")
        if body is None or body < 0:
            return add_original(builder, *args, **kwargs)
        # The pinned add_shape reads this body's list only to enumerate implicit
        # self-exclusions and append the new shape. Other bodies' lists stay in
        # place for parent/child exclusions; all bookkeeping is restored below.
        previous = builder.body_shapes[body]
        builder.body_shapes[body] = []
        try:
            return add_original(builder, *args, **kwargs)
        finally:
            added = builder.body_shapes[body]
            builder.body_shapes[body] = previous
            previous.extend(added)

    @wraps(find_original)
    def find_pairs(builder, model, *, allow_filter_blocks):
        if not allow_filter_blocks:
            return find_original(builder, model, allow_filter_blocks=allow_filter_blocks)
        model.shape_contact_pairs = None
        model.shape_contact_pair_count = 0
        setattr(model, _FLAG, True)
        print(f"NATIVE_CONTACT_STORAGE shapes={model.shape_count} explicit_candidates=deferred", flush=True)

    @wraps(solver_original)
    def initialize_solver(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        if (getattr(bound.arguments["model"], _FLAG, False)
                and not bound.arguments["use_mujoco_contacts"]):
            raise RuntimeError("Compact contact storage requires native MuJoCo contact generation")
        return solver_original(self, *args, **kwargs)

    @wraps(pipeline_original)
    def initialize_pipeline(pipeline, model, *args, **kwargs):
        if getattr(model, _FLAG, False):
            raise RuntimeError("Compact native contact storage cannot use Newton's collision pipeline")
        return pipeline_original(pipeline, model, *args, **kwargs)

    _active = True
    try:
        ModelBuilder.add_shape = add_shape
        ModelBuilder._find_shape_contact_pairs = find_pairs
        SolverMuJoCo.__init__ = initialize_solver
        SolverMuJoCo._compile_newton_collision_masks = staticmethod(compile_masks)
        SolverMuJoCo._color_collision_shapes = staticmethod(color_shapes)
        CollisionPipeline.__init__ = initialize_pipeline
        yield
    finally:
        ModelBuilder.add_shape = add_original
        ModelBuilder._find_shape_contact_pairs = find_original
        SolverMuJoCo.__init__ = solver_original
        SolverMuJoCo._compile_newton_collision_masks = masks_original
        SolverMuJoCo._color_collision_shapes = colors_original
        CollisionPipeline.__init__ = pipeline_original
        _active = False
