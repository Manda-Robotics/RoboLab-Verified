# SPDX-License-Identifier: Apache-2.0
"""Use verified compiled keyboard colliders while retaining native asset import."""
from contextlib import contextmanager
from pathlib import Path
from robolab.core.utils.newton_compiled_colliders import geometry_key, compiled_colliders
from robolab.core.utils.newton_compiled_colliders import load_colliders as _load

MESH_NAMES = frozenset(['obj_000028_Mesh'])
_active = False


def load_colliders(path):
    return _load(path, mesh_names=MESH_NAMES, max_parts=128, asset_name="keyboard")


@contextmanager
def keyboard_colliders(builder_class=None, coacd_module=None, path=None):
    global _active
    if _active:
        raise RuntimeError("Concurrent/nested keyboard collider contexts are unsupported")
    if path is None:
        path = Path(__file__).resolve().parents[3] / "assets/objects/hot3d/keyboard_colliders.usda"
    _active = True
    try:
        with compiled_colliders(builder_class, coacd_module, path, mesh_names=MESH_NAMES,
                                label_fragment='/keyboard/', max_parts=128, asset_name="keyboard") as reports:
            yield reports
    finally:
        _active = False
