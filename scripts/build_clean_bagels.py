"""Build Newton bagel USD overlays from the visible source mesh.

Requires numpy, scipy and usd-core (mujoco only for --reference-model). No simulation is advanced. The
reference MJB supplies the original Newton whole-body inertia, not contact laws.
Original files are never modified. Generated overlays retain source appearance,
deactivate old colliders, and preserve Newton whole-body mass and inertia.
Both cleaned assets use friction 5; other material properties are inherited.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
from scipy.spatial import ConvexHull


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def visible_mesh(stage):
    root = stage.GetDefaultPrim()
    cache = UsdGeom.XformCache()
    found = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh) or prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == 'invisible':
            continue
        mesh = UsdGeom.Mesh(prim)
        transform = np.asarray(cache.GetLocalToWorldTransform(prim))
        # These source assets have identity root transforms, audited explicitly.
        assert np.allclose(np.asarray(cache.GetLocalToWorldTransform(root)), np.eye(4))
        points = np.asarray(mesh.GetPointsAttr().Get()) @ transform[:3, :3] + transform[3, :3]
        assert np.all(np.asarray(mesh.GetFaceVertexCountsAttr().Get()) == 3)
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        found.append((points * 1000, faces, str(prim.GetPath())))
    assert len(found) == 1
    return found[0]


def radial_section(triangles, theta):
    """Exact visual-triangle intersections with the positive radial half-plane."""
    radial = np.array([np.cos(theta), np.sin(theta), 0.])
    normal = np.array([-np.sin(theta), np.cos(theta), 0.])
    distances = triangles @ normal
    mask = (distances.min(1) <= 0) & (distances.max(1) >= 0)
    tri, dist = triangles[mask], distances[mask]
    points = []
    for i, j in [(0, 1), (1, 2), (2, 0)]:
        hit = (dist[:, i] * dist[:, j] <= 0) & (abs(dist[:, i] - dist[:, j]) > 1e-12)
        t = dist[hit, i] / (dist[hit, i] - dist[hit, j])
        xyz = tri[hit, i] + t[:, None] * (tri[hit, j] - tri[hit, i])
        rz = np.column_stack([xyz @ radial, xyz[:, 2]])
        points.append(rz[rz[:, 0] > 0])
    return np.unique(np.round(np.concatenate(points), 9), axis=0)


def make_ring(vertices, faces, sectors=32, profile_points=24, modes=8):
    """Shared convex radial profiles, smoothly varying around the ring.

    Each hull stays within its angular wedge and includes both complete shared
    endpoint profiles. No stochastic decomposition, disconnected fragments or
    hand-set torus radius. Convexification removes small cross-section wrinkles.
    """
    count = 4 * sectors
    angles = np.arange(count) * 2 * np.pi / count
    phi = np.arange(profile_points) * 2 * np.pi / profile_points
    directions = np.column_stack([np.cos(phi), np.sin(phi)])
    profiles = []
    for theta in angles:
        points = radial_section(vertices[faces], theta)
        hull = ConvexHull(points)
        center = points[hull.vertices].mean(0)
        planes = hull.equations
        denom = directions @ planes[:, :2].T
        numer = -(planes[:, :2] @ center + planes[:, 2])
        radii = np.min(np.divide(numer, denom, out=np.full_like(denom, np.inf), where=denom > 1e-12), axis=1)
        profiles.append(center + radii[:, None] * directions)
    raw = np.array(profiles)
    coeff = np.fft.rfft(raw, axis=0)
    coeff[modes + 1:] = 0
    fitted = np.fft.irfft(coeff, n=count, axis=0)
    assert fitted[:, :, 0].min() > 3, 'No central-hole clearance'
    # Endpoint arrays are reused verbatim by adjacent hulls, including the seam.
    xyz = np.stack([fitted[:, :, 0] * np.cos(angles[:, None]),
                    fitted[:, :, 0] * np.sin(angles[:, None]), fitted[:, :, 1]], axis=2)
    xyz = (xyz / 1000).astype(np.float32).astype(np.float64) * 1000
    pieces = []
    for i in range(sectors):
        points = xyz[np.array([4*i, 4*i+2, (4*i+4) % count])].reshape(-1, 3)
        hull = ConvexHull(points)
        reduced = points[hull.vertices]
        hull = ConvexHull(reduced)
        triangles = hull.simplices.copy()
        normals = np.cross(reduced[triangles[:, 1]] - reduced[triangles[:, 0]],
                           reduced[triangles[:, 2]] - reduced[triangles[:, 0]])
        flip = np.einsum('ij,ij->i', normals, hull.equations[:, :3]) < 0
        triangles[flip] = triangles[flip][:, [0, 2, 1]]
        assert len(reduced) < 255
        pieces.append((reduced, triangles, hull.equations))
    return pieces, {'raw_profiles_rz_mm': raw, 'fitted_profiles_rz_mm': fitted,
                    'shared_profiles_xyz_mm': xyz, 'angles_rad': angles}


def build(args):
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = args.evidence_output or args.output
    evidence.mkdir(parents=True, exist_ok=True)
    assert args.sectors >= 8 and 0 <= args.modes <= 2 * args.sectors
    assert 8 <= args.profile_points <= 80
    if args.reference_model:
        import mujoco
        model = mujoco.MjModel.from_binary_path(str(args.reference_model))
        reference_hash = sha(args.reference_model)
        reference = {}
        for name in ['bagel_00', 'bagel_06']:
            body = next(i for i in range(model.nbody) if model.body(i).name.endswith('_scene_' + name))
            reference[name] = {x: getattr(model, 'body_'+x)[body].tolist() for x in ['mass', 'ipos', 'iquat', 'inertia']}
    else:
        reference_document = json.loads(args.inertial_json.read_text())
        reference_hash = reference_document['reference_geometry_model_sha256']
        reference = reference_document['assets']
    report = {'method': __doc__, 'builder_sha256': sha(__file__),
              'reference_geometry_model_sha256': reference_hash,
              'configuration': {'sectors': args.sectors, 'profile_points': args.profile_points, 'angular_fourier_modes': args.modes},
              'status': 'generated assets; dynamic validation is separate from this build',
              'sliding_friction': 5.0,
              'assets': {}}
    for name in ['bagel_00', 'bagel_06']:
        source = args.assets / (name + '.usd')
        if args.inertial_json:
            assert sha(source) == reference_document['source_asset_sha256'][name], 'Source changed: re-audit inertia and geometry'
        stage = Usd.Stage.Open(str(source))
        root = stage.GetDefaultPrim()
        vertices, faces, visual_path = visible_mesh(stage)
        pieces, fit = make_ring(vertices, faces, args.sectors, args.profile_points, args.modes)
        inertial = reference[name]
        colliders = [p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.CollisionAPI)]
        materials = {str(t) for p in colliders for t in p.GetRelationship('material:binding:physics').GetTargets()}
        assert len(colliders) == 32 and len(materials) == 1
        target = args.output / (name + '_clean.usda')
        overlay = Usd.Stage.CreateNew(str(target)) if not target.exists() else Usd.Stage.Open(str(target))
        overlay.GetRootLayer().Clear()
        overlay.GetRootLayer().subLayerPaths = [os.path.relpath(source, target.parent)]
        overlay.SetDefaultPrim(overlay.GetPrimAtPath(root.GetPath()))
        UsdGeom.SetStageMetersPerUnit(overlay, UsdGeom.GetStageMetersPerUnit(stage))
        UsdGeom.SetStageUpAxis(overlay, UsdGeom.GetStageUpAxis(stage))
        overlay.GetRootLayer().comment = 'Newton clean bagel collider; friction 5, unchanged visuals and original Newton whole-body mass/inertia. Generated by build_clean_bagels.py.'
        for old in colliders:
            overlay.OverridePrim(old.GetPath()).SetActive(False)
        api = UsdPhysics.MassAPI.Apply(overlay.OverridePrim(root.GetPath()))
        api.CreateMassAttr(inertial['mass'])
        api.CreateCenterOfMassAttr(Gf.Vec3f(*inertial['ipos']))
        api.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertial['inertia']))
        api.CreatePrincipalAxesAttr(Gf.Quatf(inertial['iquat'][0], Gf.Vec3f(*inertial['iquat'][1:])))
        api.CreateDensityAttr(0.)
        material = UsdPhysics.MaterialAPI(overlay.OverridePrim(next(iter(materials))))
        material.CreateStaticFrictionAttr(5.)
        material.CreateDynamicFrictionAttr(5.)
        data = {'visual_vertices_mm': vertices, 'visual_faces': faces, **fit}
        for i, (points, triangles, planes) in enumerate(pieces):
            prim_path = str(root.GetPath()) + f'/CleanCollisions/sector_{i:02d}'
            mesh = UsdGeom.Mesh.Define(overlay, prim_path)
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy((points / 1000).astype(np.float32)))
            mesh.CreateFaceVertexCountsAttr([3] * len(triangles))
            mesh.CreateFaceVertexIndicesAttr(triangles.ravel().tolist())
            mesh.CreateSubdivisionSchemeAttr('none')
            mesh.CreateVisibilityAttr('invisible')
            mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.array([points.min(0), points.max(0)], dtype=np.float32) / 1000))
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
            # Newton defaults to 64 vertices; preserve the already-convex authored hull.
            mesh.GetPrim().AddAppliedSchema('NewtonMeshCollisionAPI')
            limit = max(64, 1 << (len(points) - 1).bit_length())
            mesh.GetPrim().CreateAttribute('newton:maxHullVertices', Sdf.ValueTypeNames.Int, custom=False).Set(limit)
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
                UsdShade.Material(overlay.GetPrimAtPath(next(iter(materials)))), materialPurpose='physics')
            for suffix, value in [('vertices_mm', points), ('faces', triangles), ('planes', planes)]:
                data[f'piece_{i:02d}_{suffix}'] = value
        overlay.GetRootLayer().Save()
        np.savez_compressed(evidence / (name + '-geometry.npz'), **data)
        report['assets'][name] = {'source': str(source), 'source_sha256': sha(source),
                                  'overlay': target.name, 'overlay_sha256': sha(target),
                                  'visible_mesh_path': visual_path, 'original_colliders': [str(p.GetPath()) for p in colliders],
                                  'physics_material_path': next(iter(materials)), 'reference_inertial': inertial,
                                  'pieces': len(pieces), 'vertices': sum(len(p[0]) for p in pieces),
                                  'triangles': sum(len(p[1]) for p in pieces)}
        print(name, report['assets'][name]['triangles'], 'triangles', flush=True)
    (evidence / 'build-manifest.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--reference-model', type=Path)
    group.add_argument('--inertial-json', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--evidence-output', type=Path)
    p.add_argument('--sectors', type=int, default=64)
    p.add_argument('--profile-points', type=int, default=48)
    p.add_argument('--modes', type=int, default=48)
    build(p.parse_args())
