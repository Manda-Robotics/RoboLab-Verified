# KLT bin (small, visual + collision)

Copied verbatim from NVIDIA's Isaac Sim 4.5 asset library on 2026-10-27:
`https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/KLT_Bin/`.
`bananas_5_in_crate.usda` and `bottles_crate.usda` referenced that URL directly. Kit
resolves `https://` references; USD without Kit (the Kit-less OVRTX runtime) cannot,
and the crate was then missing from both scenes. Only the files the crate uses are
copied; `OmniPBR.mdl` ships with both renderers.

| File | SHA-256 |
|---|---|
| `small_KLT_visual_collision.usd` | `2e6172432c1b5ee2b7ea047229e070ccf1d49b69d01b4796c0445531af392bb9` |
| `small_KLT_visual.usd` | `2810154bdae1532dea35603024a0fefd94eaf3909cecc5edbe09b6cb510e3a2b` |
| `Materials/Textures/FOF_Map_Labels_D.png` | `7ed4642ab31d970e22c7edf62508347c6fbb14bfdf47487696a6a9a681359f22` |
| `Materials/Textures/FOF_Map_Magenta_Box_D.png` | `191e3e8f6b583dc35232f5079d18b421f97dd09d79b374b03b1b1745dd9d8477` |
| `Materials/Textures/FOF_Mesh_Labels_D.png` | `7ed4642ab31d970e22c7edf62508347c6fbb14bfdf47487696a6a9a681359f22` |

The remote MDL materials and textures in the table and kitchen fixtures still load
from the same bucket at run time. Both renderers fetch those themselves.
