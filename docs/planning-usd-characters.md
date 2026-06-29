# sims-01 — Plan: NVIDIA Reallusion USD Character Sets

Owner: Effects
Status: proposed
Last updated: 2026-06-29

Plan for supporting NVIDIA's rigged USD **character** packs (Reallusion sample
set, `Characters_NVD@10012`) in the Sim Showcase effect. These are full
production rigs — skinned, animated, blendshape-driven — and are substantially
more complex than the static `.usda` meshes in the first demo set.

---

## 1. What the pack contains

`Characters_NVD@10012` → `Assets/Characters/Reallusion/` with hero characters
(Orc, Debra, Worker) plus ActorCore variants (Business/Uniform/Party). Taking
the **Orc** as representative (`Orc/Orc.usd`, a thin ~4 KB composition layer):

| Aspect | Detail |
| --- | --- |
| Format | Binary USDC crate (pxr reads it fine; `usd-core` now installed) |
| Geometry | 11 meshes, ~36.7k verts (body 14.2k, hair 10.6k, teeth/armor/eyes/props) |
| Rig | Body fully **skinned** — 84,984 joint influences (14,164 verts × 6), 101-joint `Skeleton` |
| Animation | Baked `SkelAnimation`: 205 keyframes, timecodes 0–204 @ 60 fps (~3.4 s loop). Extra clips in `Motions/` (Idle, Slash, Shout, Roar) |
| Facial | 320 `BlendShape` prims + Audio2Face presets in the pack |
| Materials | MDL shaders + PBR texture sets (57 jpg / 49 png / tga) |
| Extras in stage | A `LightingStage` studio **backdrop** (752-unit span), `DomeLight`, `Camera` |

---

## 2. What our loader does with them today (and the gotchas)

`_load_usd_stage()` traverses all `UsdGeom.Mesh`/`UsdGeom.Points` prims, reads
`points` at default time, applies the prim's local-to-world transform, combines
everything, and normalizes to a unit cube. Against these rigs that means:

1. **Backdrop pollution.** The bundled backdrop bbox (752) is ~3× the character
   (232), so unit-normalization shrinks the Orc to a tiny blob in the corner.
2. **`rglob` over-collects.** Pointing sims at the pack root also walks every
   `Materials/*.usd` and `Motions/*.usd` (geometry-less → harmlessly skipped by
   the zero-geometry guard) plus duplicate sub-files. We want to target the
   specific character `.usd`, not the whole tree.
3. **Static bind pose.** We sample `points` at default time only — no skinning
   and no animation applied, so the character stands in raw T/A-pose.
4. **No materials.** MDL/textures ignored; rendered with our procedural palette
   shader (which is actually a fine stylized look for the viz).

Net: they *load* but currently render as a shrunken, static, palette-shaded
character against an oversized backdrop plane.

---

## 3. Tiered plan (effort vs. payoff)

### Tier 0 — Load cleanly *(small, required)*
- Restrict traversal to the character subtree: only geometry under the
  `SkelRoot` (or a configurable include/exclude prim-path list); skip cameras,
  lights, and the `LightingStage` backdrop.
- Select character files explicitly (a curated `characters/` subdir or a config
  file list) instead of rglob'ing the whole pack.
- **Result:** a clean, correctly-framed character.

### Tier 1 — Real pose, not bind pose *(small)*
- Evaluate UsdSkel skinning once at a chosen timecode via `UsdSkel.Cache` +
  `SkinningQuery.ComputeSkinnedPoints`, so the character stands posed.

### Tier 2 — Animation playback *(medium, big payoff)*
- Bake the `SkelAnimation`: sample N frames across the clip, CPU-skin each frame
  once (cached), upload as a frame sequence (sequence of VBOs, or one buffer +
  per-frame offset), and advance the frame by `self.time` / beat.
- Scaffolding already exists and is currently unused: `_SceneMesh.time_samples`
  and `current_time`, and `_load_usd_stage` already returns `time_samples`.
- Memory budget: ~36k verts × 3 floats × ~60 frames ≈ 26 MB/character — fine.
- **Result:** characters actually perform (idle → slash → roar), retimeable to
  the beat.

### Tier 3 — Audio-reactive face *(medium-high, very on-brand)*
- Drive blendshape weights (or the Audio2Face presets) from audio bands — jaw /
  mouth open on bass, expression on treble: `points + Σ wᵢ·Δᵢ`.
- **Result:** the character appears to sing / react to the track.

### Tier 4 — Textured PBR *(high, probably skip)*
- MDL is not GLSL-consumable, but the baked PBR textures are — would need UV
  extraction + a textured render path + texture loading. Big lift, and the
  stylized palette look arguably suits the viz better. Deferred.

---

## 4. Recommendation

Build **Tier 0 + Tier 2**: clean framing plus animated characters dancing to the
set. Add **Tier 3** as the follow-up flex. Keep the existing static path intact
for the simple demo meshes and the galaxy fallback.

Proposed config surface (under `[effects.SimShowcase]`):

```toml
# scene_dir / scene_dirs already exist for adding USD directories.
skel_animate     = true     # bake + play UsdSkel animation (Tier 2)
skel_frames      = 60       # frames to bake across the clip
skel_pose_time   = 0.0      # timecode for the static pose when skel_animate=false (Tier 1)
character_files  = []       # explicit character .usd list (else curated chars dir)
# Tier 3 (later):
# face_audio_react = false
```

## 5. Open questions for the owner

1. **Asset location:** curate the three hero `.usd` roots into
   `assets/sims/characters/` (committed), or wire a config that points at the
   downloaded pack in place (`~/Downloads/openusd/...`)? Committing the full
   pack is large (texture sets); a curated mesh-only copy or a path-pointer is
   lighter.
2. **Scope of first cut:** Tier 0+2 now, Tier 3 next — confirm.
3. **Frame budget:** target `skel_frames` default (60 gives smooth ~1×; higher
   = smoother + more memory/CPU bake time).

---

This is a planning document; no code changes are implied until approved.
