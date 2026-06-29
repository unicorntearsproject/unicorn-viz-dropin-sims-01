"""Tests for the GL-free USD loader (usd_scene).

These run only when OpenUSD (pxr) is installed *and* a curated character asset
is present; otherwise they skip cleanly. They never touch GL.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_US = Path(__file__).resolve().parents[1] / 'usd_scene.py'
_spec = importlib.util.spec_from_file_location('sims_usd_scene_test', _US)
usd_scene = importlib.util.module_from_spec(_spec)
# Register before exec so @dataclass(slots=True) annotation resolution (Py3.14)
# can find the module globals via sys.modules[cls.__module__].
sys.modules[_spec.name] = usd_scene
_spec.loader.exec_module(usd_scene)

pytestmark = pytest.mark.skipif(not usd_scene._USD, reason='OpenUSD (pxr) not installed')

# Curated character (git-ignored; present only on machines that downloaded it).
_REPO = Path(__file__).resolve().parents[3]
_ORC = _REPO / 'assets' / 'sims' / 'characters-nvidia' / 'Orc' / 'Orc.usd'


def _need_orc():
    if not _ORC.is_file():
        pytest.skip(f'curated character asset not present: {_ORC}')


def test_missing_stage_returns_none():
    assert usd_scene.load_scene('/nonexistent/path/none.usd') is None


def test_static_load_normalised():
    _need_orc()
    s = usd_scene.load_scene(_ORC, usd_scene.LoadOptions(animate=False))
    assert s is not None
    assert s.vert_count > 0 and len(s.indices) > 0
    # Normalised to a unit cube whose bounding box is centred at the origin
    # (vertex *mean* is offset by dense head/hand clusters, so check the bbox).
    lo, hi = s.vertices.min(0), s.vertices.max(0)
    assert (hi - lo).max() == pytest.approx(1.0, abs=1e-3)
    assert np.linalg.norm((lo + hi) * 0.5) < 1e-3
    assert s.frames is None  # static request → no frame sequence


def test_animation_bake_has_motion():
    _need_orc()
    s = usd_scene.load_scene(_ORC, usd_scene.LoadOptions(animate=True, frames=12))
    assert s is not None and s.is_animated
    assert s.frames.shape == (12, s.vert_count, 3)
    assert s.frame_normals.shape == s.frames.shape
    assert s.fps > 0
    # Some frame must differ from the first (the rig actually animates).
    motion = max(float(np.abs(f - s.frames[0]).max()) for f in s.frames)
    assert motion > 1e-4


def test_backdrop_filtered_out():
    _need_orc()
    # The bundled LightingStage backdrop (huge plane) must be excluded, else it
    # dominates normalisation and the character collapses to a tiny blob.
    s = usd_scene.load_scene(_ORC, usd_scene.LoadOptions(animate=False))
    assert s is not None
    # Character keeps a healthy extent on every axis (a lone backdrop plane
    # would leave one axis near-zero / off-centre).
    span = s.vertices.max(0) - s.vertices.min(0)
    assert (span > 0.2).all()
