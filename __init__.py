"""sims-01 drop-in package marker.

The SimShowcase effect is discovered/loaded by the core against its file path
(``sim_showcase.py``) — the hyphenated directory is not a valid module name and
``discover_dropin_effect_classes`` skips ``__init__.py`` — so this file performs
no imports. See ``sim_showcase.py`` (effect) and ``usd_scene.py`` (USD loader).
"""
