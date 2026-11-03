# Legacy engine (reference only)

The original Physio_AI_Bot monitor, unchanged except for its location. It is
kept to regenerate the behaviour baseline (`python -m tools.baseline.record_fixtures`)
and as a reference while PhysioVision still embeds it. Use the `physiosight`
package instead; see ../docs/BEHAVIOR_CHANGES.md for every difference.

`config.py` resolves the models relative to this folder (`legacy/models/`),
which does not exist; to run the legacy monitor directly, link or copy
`../models` here first. Normally use `physiosight run` instead.
