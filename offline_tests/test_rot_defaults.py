"""Every RoboLab-authored config rotation is explicit (robolab/core/utils/rot_defaults.py): an
inherited class default is Isaac Lab's XYZW identity on Isaac Lab 3, which the WXYZ boundary
conversion turns into a 180-degree Z rotation (docs/isaac_sim_6.md). Static, no simulator."""
import glob
import os

from robolab.core.utils.rot_defaults import missing_rotations

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = sorted(glob.glob(os.path.join(ROOT, "robolab", "robots", "*.py"))
               + glob.glob(os.path.join(ROOT, "robolab", "variations", "*.py"))
               + glob.glob(os.path.join(ROOT, "robolab", "registrations", "**", "*.py"), recursive=True)
               + [os.path.join(ROOT, "robolab", "core", "environments", "base.py")])


def test_every_config_rotation_is_explicit():
    assert FILES
    problems = {os.path.relpath(f, ROOT): missing_rotations(open(f).read()) for f in FILES}
    problems = {k: v for k, v in problems.items() if v}
    assert not problems, "\n".join(f"{k}: {v}" for k, v in problems.items())


def test_checker_catches_the_original_defects():
    src = '''
    dome_light = AssetBaseCfg(prim_path="/World/background", spawn=x)
    frames = FrameTransformerCfg(prim_path="{ENV_REGEX_NS}/robot/panda_link0",
        target_frames=[FrameTransformerCfg.FrameCfg(prim_path="p", name="n")])
    body_offset = DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0])
    init_state = AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.6, 0.7))
    '''
    p = missing_rotations(src)
    assert any("FrameCfg without offset" in x for x in p)
    assert any("FrameTransformerCfg without source_frame_offset" in x for x in p)
    assert any("OffsetCfg without rot" in x for x in p)
    assert any("InitialStateCfg without rot" in x for x in p)
    fixed = src.replace("OffsetCfg(pos=", "OffsetCfg(rot=(1.0, 0.0, 0.0, 0.0), pos=").replace("InitialStateCfg(pos=", "InitialStateCfg(rot=(1.0, 0.0, 0.0, 0.0), pos=") \
               .replace('name="n")', 'name="n", offset=OffsetCfg(rot=(1.0, 0.0, 0.0, 0.0)))').replace('panda_link0",', 'panda_link0", source_frame_offset=OffsetCfg(rot=(1.0, 0.0, 0.0, 0.0)),')
    assert missing_rotations(fixed) == []
