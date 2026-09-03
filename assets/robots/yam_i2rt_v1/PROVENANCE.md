# I2RT YAM v1 model files (source for `bimanual_yam`)

Copied verbatim from https://github.com/i2rt-robotics/i2rt (MIT License, Copyright I2RT Robotics),
path `i2rt/robot_models/arm/yam/v1/`, on 2026-09-02:

- `yam_linear_4310_d405.urdf` / `.xml` — YAM v1 arm + linear_4310 gripper + D405 wrist bracket
- `yam.urdf` — arm only; `README.md` — I2RT's physical-properties sheet
- `assets/*.stl`, `assets/d405/*.stl` — meshes

`_utils/build_bimanual_yam.py` turns the URDF into `../bimanual_yam/bimanual_yam.usd`.
