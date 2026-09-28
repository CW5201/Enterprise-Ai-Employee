"""Enterprise task generator (PLACEHOLDER).

Status
------
Phase 0 (project skeleton). No generation logic is implemented.

Purpose (Phase 5)
-----------------
Generate the ~430 enterprise tasks described in docs/DATASET.md by combining
the WideWorldImporters / AdventureWorks schema with the public enterprise
policy corpus, organised by category:

    information_query / knowledge_qa / text_to_sql /
    data_analysis / multi_source_reasoning / complex_agent_task

Hard rules for whoever implements this
--------------------------------------
1. Tasks are generated from declared rules, never hand-fabricated.
2. Ground truth for SQL tasks must be produced by actually executing the query
   (data/synthetic/generate_ground_truth.py), never written by hand.
3. Tasks derived from third-party benchmarks (Spider / BIRD / HotpotQA /
   BFCL / ToolBench / GAIA) keep their original licence and attribution.
4. Any task used as a few-shot example must be excluded from the eval set.

TODO(Phase 5): implement.
"""

from __future__ import annotations

__all__: list[str] = []

# TODO(Phase 5): implement according to docs/ROADMAP.md.
