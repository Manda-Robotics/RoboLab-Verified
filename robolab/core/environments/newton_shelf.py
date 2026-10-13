# SPDX-License-Identifier: Apache-2.0
"""Use verified compiled shelf colliders while retaining native asset import."""
from contextlib import contextmanager
from pathlib import Path
from robolab.core.utils.newton_compiled_colliders import geometry_key, compiled_colliders
from robolab.core.utils.newton_compiled_colliders import load_colliders as _load

MESH_NAMES = frozenset(['M_Rack_L01_Frame', 'M_Rack_L01_FrameJoints', 'M_Rack_L01_Shelf', 'M_Rack_L01_ShelfElements'])
_active = False


def load_colliders(path):
    return _load(path, mesh_names=MESH_NAMES, max_parts=64, asset_name="shelf")


@contextmanager
def shelf_colliders(builder_class=None, coacd_module=None, path=None):
    global _active
    if _active:
        raise RuntimeError("Concurrent/nested shelf collider contexts are unsupported")
    if path is None:
        path = Path(__file__).resolve().parents[3] / "assets/objects/vomp/rack_l04/rack_l04_colliders.usda"
    _active = True
    try:
        with compiled_colliders(builder_class, coacd_module, path, mesh_names=MESH_NAMES,
                                label_fragment='/SM_Rack_L04_01/', max_parts=64, asset_name="shelf") as reports:
            yield reports
    finally:
        _active = False
