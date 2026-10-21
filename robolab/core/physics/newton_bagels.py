# SPDX-License-Identifier: Apache-2.0
"""Select clean bagel assets for Newton without changing shared task scenes."""
from pathlib import Path


SCENE_OVERLAYS = {
    "bagel_plate_banana_bowl.usda": "bagel_plate_banana_bowl_clean.usda",
    "breakfast_table.usda": "breakfast_table_clean.usda",
    "test_plate_banana_rubiks_cube.usda": "test_plate_banana_rubiks_cube_clean.usda",
}
SCENE_DIR = Path(__file__).resolve().parents[3] / "assets/scenes"


def configure_scene(cfg):
    """Replace bundled bagel scenes before spawning; return whether one is used.

    Match full paths so a custom scene with the same filename is not replaced.
    The overlays author friction 5 on the two cleaned assets. Explicit runtime
    friction requests still take precedence through the usual Newton adapter.
    """
    spawn = getattr(getattr(cfg.scene, "scene", None), "spawn", None)
    source = getattr(spawn, "usd_path", None)
    if not source:
        return False
    source = Path(source).resolve()
    for original, overlay in SCENE_OVERLAYS.items():
        target = SCENE_DIR / overlay
        if source not in {(SCENE_DIR / original).resolve(), target.resolve()}:
            continue
        if not target.is_file():
            raise FileNotFoundError(f"Newton bagel scene overlay missing: {target}")
        spawn.usd_path = str(target)
        cfg.newton_bagel_assets = {
            "scene": overlay,
            "assets": ["bagel_00_clean.usda", "bagel_06_clean.usda"],
            "convex_sectors_per_asset": 64,
            "authored_sliding_friction": 5.0,
        }
        return True
    return False
