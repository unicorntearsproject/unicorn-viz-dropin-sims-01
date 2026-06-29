"""Fetch + convert real open-source robots (MuJoCo Menagerie) to animated USD.

Downloads a model's MJCF + OBJ meshes from the Apache-2.0 licensed
``google-deepmind/mujoco_menagerie`` repo and converts it to a USD stage whose
joints gently sweep, so it animates through sims-01's rigid joint-Xform path.

Output goes to ``assets/sims/robots-menagerie/<model>.usda`` (git-ignored — see
assets/sims/README.md). Menagerie is Apache-2.0; a CREDITS.txt with per-model
attribution is written alongside. Run:

    python drop-ins/sims-01/tools/fetch_robots.py

Requires OpenUSD + network access.
"""
from __future__ import annotations

import math
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, Gf, Vt

_RAW = 'https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main'
_OUT = Path(__file__).resolve().parents[3] / 'assets' / 'sims' / 'robots-menagerie'
_FPS = 30.0
_FRAMES = 120

# (model dir, main xml, list of body names to start joint-sweep phase variety)
_MODELS = [
    ('universal_robots_ur5e', 'ur5e.xml'),
    ('unitree_go2', 'go2.xml'),
]


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 - fixed host
        return r.read()


# --------------------------------------------------------------------------- OBJ
def _parse_obj(data: bytes):
    verts: list[tuple[float, float, float]] = []
    counts: list[int] = []
    idx: list[int] = []
    for line in data.decode('utf-8', 'ignore').splitlines():
        if line.startswith('v '):
            _, x, y, z = line.split()[:4]
            verts.append((float(x), float(y), float(z)))
        elif line.startswith('f '):
            f = [int(tok.split('/')[0]) - 1 for tok in line.split()[1:]]
            if len(f) >= 3:
                counts.append(len(f))
                idx.extend(f)
    return verts, counts, idx


# ------------------------------------------------------------------ MJCF helpers
def _floats(s: str | None, default):
    if not s:
        return list(default)
    return [float(x) for x in s.split()]


def _quat_to_mat(q):  # MJCF quat is (w, x, y, z)
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0],
        [0, 0, 0, 1]], dtype=np.float64)


def _axis_angle_mat(axis, ang):
    ax = np.asarray(axis, float)
    ax = ax / (np.linalg.norm(ax) or 1.0)
    x, y, z = ax
    c, s, t = math.cos(ang), math.sin(ang), 1 - math.cos(ang)
    return np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x, 0],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0],
        [0, 0, 0, 1]], dtype=np.float64)


def _trans_mat(p):
    m = np.eye(4)
    m[:3, 3] = p
    return m


def _resolve_joint_axes(root: ET.Element) -> dict:
    """Map default-class name -> joint axis, resolving childclass inheritance."""
    axes: dict[str, list[float]] = {}

    def walk(elem, inherited):
        cls = elem.get('class')
        axis = inherited
        j = elem.find('joint')
        if j is not None and j.get('axis'):
            axis = _floats(j.get('axis'), axis)
        if cls:
            axes[cls] = axis
        for child in elem.findall('default'):
            walk(child, axis)

    for d in root.findall('default'):
        walk(d, [0.0, 1.0, 0.0])
    return axes


# --------------------------------------------------------------------- conversion
def convert(model_dir: str, xml_name: str) -> Path | None:
    xml = _get(f'{_RAW}/{model_dir}/{xml_name}')
    root = ET.fromstring(xml)
    meshdir = (root.find('compiler').get('meshdir', 'assets')
               if root.find('compiler') is not None else 'assets')

    # asset meshes: name -> filename
    mesh_files: dict[str, str] = {}
    for am in root.iter('mesh'):
        fn = am.get('file')
        if fn:
            mesh_files[am.get('name') or Path(fn).stem] = fn

    # download + parse OBJ meshes (only ones actually referenced are needed, but
    # grabbing all is simplest and small)
    meshes: dict[str, tuple] = {}
    for name, fn in mesh_files.items():
        if not fn.lower().endswith('.obj'):
            continue
        try:
            meshes[name] = _parse_obj(_get(f'{_RAW}/{model_dir}/{meshdir}/{fn}'))
        except Exception as exc:
            print(f'  skip mesh {fn}: {exc}')

    axes = _resolve_joint_axes(root)

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)   # MuJoCo is Z-up
    stage.SetStartTimeCode(0.0)
    stage.SetEndTimeCode(float(_FRAMES))
    stage.SetTimeCodesPerSecond(_FPS)
    safe = model_dir.replace('-', '_')
    rootx = UsdGeom.Xform.Define(stage, f'/{safe}')
    stage.SetDefaultPrim(rootx.GetPrim())

    worldbody = root.find('worldbody')
    joint_n = [0]

    def add_body(body_el, parent_path, childclass):
        cc = body_el.get('childclass', childclass)
        name = (body_el.get('name') or f'body{id(body_el) % 10000}').replace('-', '_')
        path = f'{parent_path}/{name}'
        bx = UsdGeom.Xform.Define(stage, path)
        T = _trans_mat(_floats(body_el.get('pos'), [0, 0, 0])) @ \
            _quat_to_mat(_floats(body_el.get('quat'), [1, 0, 0, 0]))

        joint = body_el.find('joint')
        if joint is not None and joint.get('type', 'hinge') in ('hinge', None):
            jcls = joint.get('class', cc)
            axis = _floats(joint.get('axis'), axes.get(jcls, [0, 1, 0]))
            jpos = _floats(joint.get('pos'), [0, 0, 0])
            phase = joint_n[0] * 0.7
            joint_n[0] += 1
            amp = math.radians(35.0)
            op = bx.AddTransformOp()
            for f in range(_FRAMES + 1):
                ang = amp * math.sin(_TAU * f / _FRAMES + phase)
                R = _trans_mat(jpos) @ _axis_angle_mat(axis, ang) @ _trans_mat(
                    [-c for c in jpos])
                op.Set(Gf.Matrix4d(*(T @ R).flatten()), Usd.TimeCode(float(f)))
        else:
            bx.AddTransformOp().Set(Gf.Matrix4d(*T.flatten()))

        gi = 0
        for geom in body_el.findall('geom'):
            mname = geom.get('mesh')
            if mname is None or mname not in meshes:
                continue
            verts, counts, idx = meshes[mname]
            if not verts:
                continue
            gT = _trans_mat(_floats(geom.get('pos'), [0, 0, 0])) @ \
                _quat_to_mat(_floats(geom.get('quat'), [1, 0, 0, 0]))
            pts = np.asarray(verts, float)
            homo = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
            wp = (homo @ gT.T)[:, :3]
            m = UsdGeom.Mesh.Define(stage, f'{path}/geom{gi}')
            m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(wp.astype(np.float32)))
            m.CreateFaceVertexCountsAttr(counts)
            m.CreateFaceVertexIndicesAttr(idx)
            m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            gi += 1

        for child in body_el.findall('body'):
            add_body(child, path, cc)

    for b in worldbody.findall('body'):
        add_body(b, f'/{safe}', None)

    _OUT.mkdir(parents=True, exist_ok=True)
    out = _OUT / f'{model_dir}.usda'
    stage.GetRootLayer().Export(str(out))
    return out


_TAU = 2.0 * math.pi


if __name__ == '__main__':
    _OUT.mkdir(parents=True, exist_ok=True)
    credits = ['Robots converted from MuJoCo Menagerie '
               '(https://github.com/google-deepmind/mujoco_menagerie),',
               'licensed under Apache-2.0. Each model retains its upstream '
               'LICENSE; see the per-model directories upstream.', '']
    for md, xml in _MODELS:
        try:
            out = convert(md, xml)
            print('wrote', out)
            credits.append(f'- {md}: {_RAW}/{md}')
        except Exception as exc:
            print(f'FAILED {md}: {exc}')
    (_OUT / 'CREDITS.txt').write_text('\n'.join(credits) + '\n')
