# Data layout

- `raw/`: small authoritative source tables retained without model-side edits.
- `processed/`: model-ready tables derived from raw sources. These are not
  hand-edited source data.
- `external/`: canonical large artifacts obtained from a versioned public
  archive. The Phase 2 bootstrap will populate this ignored directory.
- `generated/`: reproducible intermediate files. This ignored directory is not
  an authoritative source.

Large hourly tables still tracked after Phase 1 are Phase 2 externalization
candidates. They were not moved or regenerated in this phase.
