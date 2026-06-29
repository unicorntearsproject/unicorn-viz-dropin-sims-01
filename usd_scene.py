"""Reusable, GL-free USD scene loader for sims-01.

Turns a USD stage into plain numpy arrays the renderer can upload, handling the
three classes of content we care about across NVIDIA's packs:

- **Static meshes** (the simple demo set): points + faces at the default time.
- **Skinned, animated characters** (Reallusion/ActorCore): UsdSkel rigs with a
  bound `SkelAnimation`. Geometry is *baked* — skinned vertex positions are
  evaluated on the CPU at N sampled times into a frame sequence the renderer
  plays back. Robust to packs that bind the animation on the ``Skeleton`` prim
  (the cache's auto-resolution misses those, so we build the AnimQuery and
  skinning transforms manually).
- **Blendshapes** (facial / Audio2Face): a handful of audio-mappable targets
  (jaw/mouth/…) are extracted as per-vertex deltas so the runtime can weight
  them from the audio signal.

This module is deliberately free of ``moderngl`` so it can be unit-tested (and
reused by future packs) without a GL context. The effect owns all GL upload.

All output geometry is normalised to a unit cube centred at the origin; the same
centre/scale is applied to every animation frame and blendshape delta so they
stay registered with the base mesh.

Designed to be pack-agnostic: tune behaviour through :class:`LoadOptions` rather
than hard-coding any single pack's prim layout.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

try:
    from pxr import Usd, UsdGeom, UsdSkel, Vt  # type: ignore
    _USD = True
except ImportError:  # pragma: no cover - exercised only without OpenUSD
    _USD = False


# Prim types that never contribute renderable geometry for our purposes.
_DEFAULT_SKIP_TYPES = frozenset({
    'Camera', 'DomeLight', 'RectLight', 'DiskLight', 'SphereLight',
    'DistantLight', 'CylinderLight', 'GeometryLight', 'PortalLight',
    'PluginLight',
})
# Path keywords that flag bundled studio scaffolding (backdrops, light rigs).
_DEFAULT_SKIP_KEYWORDS = ('lightingstage', 'backdrop', 'background', '/env/',
                          'environment', 'groundplane')
# Blendshape name fragments worth wiring to audio (ARKit / Audio2Face / CC).
_DEFAULT_BLEND_HINTS = ('jaw', 'mouth', 'open', 'a25', 'aa', 'smile')


@dataclass(slots=True)
class LoadOptions:
    """Tunables for :func:`load_scene` — pack-agnostic, with safe defaults."""

    animate: bool = True
    frames: int = 48                 # baked frames across the animation clip
    pose_time: float | None = None   # static pose timecode when not animating
    blendshapes: bool = False        # extract audio-mappable blendshape deltas
    blend_hints: tuple[str, ...] = _DEFAULT_BLEND_HINTS
    max_blend_targets: int = 4
    skip_prim_types: frozenset[str] = _DEFAULT_SKIP_TYPES
    skip_path_keywords: tuple[str, ...] = _DEFAULT_SKIP_KEYWORDS


@dataclass(slots=True)
class LoadedScene:
    """Normalised geometry ready for GPU upload.

    ``frames`` is ``None`` for static scenes; otherwise shape ``(F, V, 3)`` with
    matching ``frame_normals``. ``blend_deltas`` is ``None`` unless blendshapes
    were requested and found, else shape ``(B, V, 3)`` registered to the base
    vertex order (zero outside the affected mesh).
    """

    name: str
    vertices: np.ndarray            # (V, 3) f32 — base frame (frame 0)
    normals: np.ndarray             # (V, 3) f32
    indices: np.ndarray             # (T, 3) u32
    points: np.ndarray              # (P, 3) f32 — loose point cloud (no faces)
    frames: np.ndarray | None = None        # (F, V, 3) f32
    frame_normals: np.ndarray | None = None  # (F, V, 3) f32
    fps: float = 30.0
    blend_deltas: np.ndarray | None = None   # (B, V, 3) f32
    blend_names: list[str] = field(default_factory=list)

    @property
    def is_animated(self) -> bool:
        return self.frames is not None and len(self.frames) > 1

    @property
    def vert_count(self) -> int:
        return int(len(self.vertices))

    @property
    def point_count(self) -> int:
        return int(len(self.points))


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _zup_to_yup(arr: np.ndarray) -> np.ndarray:
    """Rotate Z-up coordinates/vectors into the renderer's Y-up convention.

    USD stages may be authored Z-up (NVIDIA character packs) or Y-up (our demo
    scenes); the renderer is Y-up. Maps (x, y, z) -> (x, z, -y), a -90° rotation
    about X. Works for positions and (direction) normals alike. Operates on the
    last axis, so it handles (N,3) and (F,V,3) arrays.
    """
    out = arr[..., [0, 2, 1]].copy()
    out[..., 2] *= -1.0
    return out


def _compute_normals(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals for a triangle mesh."""
    normals = np.zeros_like(verts)
    if len(tris) == 0:
        normals[:, 2] = 1.0
        return normals
    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    for i in range(3):
        np.add.at(normals, tris[:, i], fn)
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    lens[lens == 0.0] = 1.0
    return (normals / lens).astype(np.float32)


def _triangulate(face_counts: np.ndarray, face_indices: np.ndarray,
                 vert_offset: int) -> np.ndarray:
    """Fan-triangulate arbitrary polygons into a ``(T, 3)`` index array."""
    tris: list[tuple[int, int, int]] = []
    cursor = 0
    for count in face_counts:
        if count >= 3:
            base = face_indices[cursor]
            for k in range(1, count - 1):
                tris.append((face_indices[cursor + k],
                             face_indices[cursor + k + 1], base))
        cursor += count
    if not tris:
        return np.zeros((0, 3), dtype=np.uint32)
    return np.asarray(tris, dtype=np.uint32) + vert_offset


# ---------------------------------------------------------------------------
# Stage inspection
# ---------------------------------------------------------------------------

def _is_skipped(prim, opts: LoadOptions) -> bool:
    if prim.GetTypeName() in opts.skip_prim_types:
        return True
    low = prim.GetPath().pathString.lower()
    return any(k in low for k in opts.skip_path_keywords)


def _local_to_world(prim) -> np.ndarray:
    xf = np.eye(4, dtype=np.float32)
    try:
        gf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default())
        for r in range(4):
            for c in range(4):
                xf[r, c] = gf[r][c]
    except Exception:
        pass
    return xf


def _mesh_points_world(prim, time) -> np.ndarray | None:
    mesh = UsdGeom.Mesh(prim)
    raw = mesh.GetPointsAttr().Get(time)
    if raw is None:
        ts = mesh.GetPointsAttr().GetTimeSamples()
        raw = mesh.GetPointsAttr().Get(ts[0]) if ts else None
    if raw is None or len(raw) == 0:
        return None
    pts = np.asarray(raw, dtype=np.float32)
    xf = _local_to_world(prim)
    homo = np.concatenate([pts, np.ones((len(pts), 1), np.float32)], axis=1)
    return (homo @ xf.T)[:, :3]


def _mesh_topology(prim, vert_offset: int) -> np.ndarray:
    mesh = UsdGeom.Mesh(prim)
    fc = mesh.GetFaceVertexCountsAttr().Get()
    fi = mesh.GetFaceVertexIndicesAttr().Get()
    if fc is None or fi is None:
        return np.zeros((0, 3), dtype=np.uint32)
    return _triangulate(np.asarray(fc, np.int32),
                        np.asarray(fi, np.int32), vert_offset)


# ---------------------------------------------------------------------------
# Skeletal animation (manual binding — robust across packs)
# ---------------------------------------------------------------------------

class _SkelBinder:
    """Wraps a resolved skeleton query, producing per-joint skinning transforms
    via UsdSkel's authoritative :meth:`ComputeSkinningTransforms` (handles the
    bind/rest/anim-space conventions correctly)."""

    def __init__(self, skel_query):
        self._sq = skel_query

    def skinning_transforms(self, time):
        return self._sq.ComputeSkinningTransforms(time)


def _animation_source(skel_prim):
    """Return the SkelAnimation prim bound to a skeleton, or ``None``."""
    rel = UsdSkel.BindingAPI(skel_prim).GetAnimationSourceRel()
    targets = rel.GetTargets() if rel else []
    if not targets:
        return None
    return skel_prim.GetStage().GetPrimAtPath(targets[0])


def _ancestor_skel_root(prim):
    """Return the nearest SkelRoot ancestor of ``prim`` (inclusive), or None."""
    cur = prim
    while cur and cur.IsValid():
        if cur.IsA(UsdSkel.Root):
            return cur
        cur = cur.GetParent()
    return None


# ---------------------------------------------------------------------------
# Blendshapes (Tier 3 — audio-mappable facial deltas)
# ---------------------------------------------------------------------------

def _extract_blend_deltas(prim, n_local_verts: int, opts: LoadOptions,
                          ) -> tuple[list[np.ndarray], list[str]]:
    """Mesh-local per-vertex deltas for blendshapes matching the name hints.

    Returns ``(deltas, names)`` where each delta is shape ``(n_local_verts, 3)``
    in the mesh's own vertex order; the caller places it at the mesh's offset in
    the combined buffer.
    """
    binding = UsdSkel.BindingAPI(prim)
    names = binding.GetBlendShapesAttr().Get()
    targets = binding.GetBlendShapeTargetsRel().GetTargets() \
        if binding.GetBlendShapeTargetsRel() else []
    if not names or not targets:
        return [], []
    stage = prim.GetStage()
    deltas: list[np.ndarray] = []
    chosen: list[str] = []
    for name, tgt in zip(names, targets):
        if len(chosen) >= opts.max_blend_targets:
            break
        if not any(h in str(name).lower() for h in opts.blend_hints):
            continue
        bs = stage.GetPrimAtPath(tgt)
        if not bs:
            continue
        offsets = bs.GetAttribute('offsets').Get()
        idx = bs.GetAttribute('pointIndices').Get()
        if offsets is None:
            continue
        local = np.zeros((n_local_verts, 3), dtype=np.float32)
        off = np.asarray(offsets, dtype=np.float32)
        if idx is not None:
            pidx = np.asarray(idx, dtype=np.int64)
        else:
            pidx = np.arange(min(len(off), n_local_verts), dtype=np.int64)
        n = min(len(off), len(pidx))
        valid = pidx[:n] < n_local_verts
        local[pidx[:n][valid]] = off[:n][valid]
        deltas.append(local)
        chosen.append(str(name))
    return deltas, chosen


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_scene(path, options: LoadOptions | None = None) -> LoadedScene | None:
    """Load a USD stage into a normalised :class:`LoadedScene`.

    Returns ``None`` when OpenUSD is unavailable, the stage fails to open, or no
    renderable geometry is found (the caller should then fall back).
    """
    opts = options or LoadOptions()
    if not _USD:
        return None
    try:
        stage = Usd.Stage.Open(str(path))
    except Exception as exc:
        log.warning('usd_scene: failed to open %s: %s', path, exc)
        return None
    if not stage:
        log.warning('usd_scene: failed to open %s', path)
        return None

    name = stage.GetDefaultPrim().GetName() if stage.GetDefaultPrim() \
        else getattr(path, 'stem', str(path))
    fps = stage.GetTimeCodesPerSecond() or 30.0

    # --- Resolve skeleton binding (if any) ---------------------------------
    # Mirror each skeleton's bound animation onto its SkelRoot (in-memory): the
    # NVIDIA packs author skel:animationSource on the Skeleton prim, which
    # UsdSkelCache does not auto-resolve, so the high-level skinning API would
    # otherwise see no animation. Authoring it on the SkelRoot makes the cache
    # resolve it and ComputeSkinningTransforms returns the correct pose.
    anim_times: list[float] = []
    for skel_prim in (p for p in stage.Traverse() if p.IsA(UsdSkel.Skeleton)):
        anim_prim = _animation_source(skel_prim)
        sroot = _ancestor_skel_root(skel_prim)
        if anim_prim is None or sroot is None:
            continue
        rb = UsdSkel.BindingAPI.Apply(sroot)
        if not (rb.GetAnimationSourceRel()
                and rb.GetAnimationSourceRel().GetTargets()):
            rb.CreateAnimationSourceRel().SetTargets([anim_prim.GetPath()])
        # Collect keyframe times so we can recover the clip's range when the
        # stage itself authors no start/end timecodes (motion-variant files do).
        try:
            anim_times.extend(
                UsdSkel.Animation(anim_prim).GetRotationsAttr().GetTimeSamples())
        except Exception:
            pass

    cache = UsdSkel.Cache()
    binders: dict[str, _SkelBinder] = {}   # skinnable prim path -> binder
    skinning_query: dict[str, object] = {}
    roots = [p for p in stage.Traverse() if p.IsA(UsdSkel.Root)]
    for root in roots:
        cache.Populate(UsdSkel.Root(root), Usd.TraverseInstanceProxies())
    for skel_prim in (p for p in stage.Traverse() if p.IsA(UsdSkel.Skeleton)):
        skel_query = cache.GetSkelQuery(UsdSkel.Skeleton(skel_prim))
        if not skel_query or not skel_query.GetAnimQuery():
            continue
        binder = _SkelBinder(skel_query)
        for child in Usd.PrimRange(skel_prim.GetParent()):
            sq = cache.GetSkinningQuery(child)
            if sq:
                binders[child.GetPath().pathString] = binder
                skinning_query[child.GetPath().pathString] = sq

    # --- Time sampling -----------------------------------------------------
    start, end = stage.GetStartTimeCode(), stage.GetEndTimeCode()
    if end <= start and anim_times:
        # Motion-variant clips often omit stage timecodes; use the keyframes.
        start, end = min(anim_times), max(anim_times)
    has_anim = bool(binders) and end > start and opts.animate
    if has_anim:
        n = max(2, int(opts.frames))
        times = [start + (end - start) * i / (n - 1) for i in range(n)]
    else:
        t = opts.pose_time if opts.pose_time is not None else start
        times = [t]

    # --- Gather meshes (Tier 0 filtering) ----------------------------------
    mesh_prims = [p for p in stage.Traverse()
                  if p.IsA(UsdGeom.Mesh) and not _is_skipped(p, opts)]

    # Per-frame vertex blocks, concatenated in a stable prim order.
    frame_blocks: list[list[np.ndarray]] = [[] for _ in times]
    tri_list: list[np.ndarray] = []
    loose_points: list[np.ndarray] = []
    blend_deltas: list[np.ndarray] = []
    blend_names: list[str] = []
    vert_offset = 0

    for prim in mesh_prims:
        ppath = prim.GetPath().pathString
        binder = binders.get(ppath)
        sq = skinning_query.get(ppath)
        rest = _mesh_points_world(prim, Usd.TimeCode(times[0]))
        if rest is None:
            continue
        n_local = len(rest)

        if binder is not None and sq is not None and has_anim:
            mesh = UsdGeom.Mesh(prim)
            for fi, t in enumerate(times):
                tc = Usd.TimeCode(t)
                raw = mesh.GetPointsAttr().Get(tc)
                pts = Vt.Vec3fArray(raw) if raw is not None \
                    else Vt.Vec3fArray(list(mesh.GetPointsAttr().Get()))
                sq.ComputeSkinnedPoints(binder.skinning_transforms(tc), pts, tc)
                frame_blocks[fi].append(np.asarray(pts, dtype=np.float32))
        else:
            # Static (or skinned-but-not-animating): constant across frames.
            for fi in range(len(times)):
                frame_blocks[fi].append(rest)

        tri_list.append(_mesh_topology(prim, vert_offset))

        if opts.blendshapes:
            # Store mesh-local deltas; placed into the combined buffer below.
            d, nm = _extract_blend_deltas(prim, n_local, opts)
            for delta_local, name in zip(d, nm):
                blend_deltas.append((vert_offset, delta_local))  # type: ignore
                blend_names.append(name)
        vert_offset += n_local

    # --- Loose points (point clouds, e.g. simple packs) --------------------
    for prim in (p for p in stage.Traverse()
                 if p.IsA(UsdGeom.Points) and not _is_skipped(p, opts)):
        raw = UsdGeom.Points(prim).GetPointsAttr().Get(Usd.TimeCode(times[0]))
        if raw is not None and len(raw) > 0:
            loose_points.append(np.asarray(raw, dtype=np.float32))

    if vert_offset == 0 and not loose_points:
        return None

    # --- Assemble frames + normalise ---------------------------------------
    frames_np: np.ndarray | None = None
    if vert_offset > 0:
        frames_np = np.stack([np.concatenate(block, axis=0)
                              for block in frame_blocks], axis=0)  # (F, V, 3)
    indices = np.concatenate(tri_list, axis=0) if tri_list \
        else np.zeros((0, 3), dtype=np.uint32)
    points = np.concatenate(loose_points, axis=0) if loose_points \
        else np.zeros((0, 3), dtype=np.float32)

    # Convert Z-up stages into the renderer's Y-up before normalising so normals
    # (computed below from these positions) come out correct automatically.
    if UsdGeom.GetStageUpAxis(stage) == 'Z':
        if frames_np is not None:
            frames_np = _zup_to_yup(frames_np)
        if len(points):
            points = _zup_to_yup(points)
        blend_deltas = [(off, _zup_to_yup(local)) for off, local in blend_deltas]

    # Normalise by the base frame's bbox; apply to every frame + points.
    ref = frames_np[0] if frames_np is not None else points
    centre = (ref.min(0) + ref.max(0)) * 0.5
    scale = float(np.max(ref.max(0) - ref.min(0))) + 1e-9
    if frames_np is not None:
        frames_np = (frames_np - centre) / scale
    if len(points):
        points = (points - centre) / scale

    base = frames_np[0] if frames_np is not None \
        else np.zeros((0, 3), dtype=np.float32)
    base_normals = _compute_normals(base, indices) if len(base) else \
        np.zeros((0, 3), dtype=np.float32)

    frame_normals_np: np.ndarray | None = None
    out_frames: np.ndarray | None = None
    if frames_np is not None and len(frames_np) > 1:
        out_frames = frames_np
        frame_normals_np = np.stack(
            [_compute_normals(f, indices) for f in frames_np], axis=0)

    # Blendshape deltas: scale by normalisation, expand to (B, V, 3).
    deltas_np: np.ndarray | None = None
    if blend_deltas and frames_np is not None:
        total_v = frames_np.shape[1]
        stacked = []
        for off, local in blend_deltas:
            dense = np.zeros((total_v, 3), dtype=np.float32)
            m = min(len(local), total_v - off)
            dense[off:off + m] = local[:m] / scale
            stacked.append(dense)
        deltas_np = np.stack(stacked, axis=0)

    return LoadedScene(
        name=name,
        vertices=base,
        normals=base_normals,
        indices=indices,
        points=points,
        frames=out_frames,
        frame_normals=frame_normals_np,
        fps=float(fps),
        blend_deltas=deltas_np,
        blend_names=blend_names,
    )
