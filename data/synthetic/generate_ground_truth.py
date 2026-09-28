"""Ground truth / evidence generator (PLACEHOLDER).

Status
------
Phase 0 (project skeleton). No generation logic is implemented.

Purpose (Phase 5)
-----------------
Produce, for every task in data/eval_dataset.jsonl, the fields:

    ground_truth   - the reference answer
    evidence       - the evidence items that support it

Rules
-----
- SQL tasks: ground truth comes from executing the reference query against the
  DuckDB database. It is never written by hand.
- Knowledge tasks: ground truth and evidence come from the source document
  (doc_id + chunk id + span), so citations can be checked mechanically.
- Multi-source tasks: every evidence item records its origin
  (milvus chunk / sql result set / neo4j path / analysis output).
- Evidence ids must be stable across reruns, otherwise claim-evidence
  verification cannot be evaluated reproducibly.

TODO(Phase 5): implement.
"""

from __future__ import annotations

__all__: list[str] = []

# TODO(Phase 5): implement according to docs/ROADMAP.md.
