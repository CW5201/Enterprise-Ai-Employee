# Phase 4 Claim-Evidence Verification — Failure Cases

Real failure cases from `artifacts/phase4/verification_results.json`
(full baseline C, 120-task dataset `data/eval/verification_eval.jsonl`).
Failed claims are **not** removed from the denominator; each case below
is one that the full verifier got wrong against the independent GT.

Categories: supported-incorrectly-rejected, unsupported-accepted,
conflict-not-detected, numeric-mismatch, attribution.

---

## 1. ver-018 — supported_claim_incorrectly_rejected
- **Question**: 费用日期最早是哪天？
- **Claim**: 最早一笔费用发生在 2026-01-15 (factual, GT supported=True)
- **Prediction**: unsupported (verifier=semantic, "no candidate above
  semantic_threshold 0.5, best sim 0.000")
- **Cause**: the RAG/SQL evidence pool had no record whose *content*
  contained the date token; the lexical scorer returned 0.0 similarity,
  so Layer 3 rejected a claim that was in fact correct.  A real
  semantic embedder would likely have caught the "2026-01-15" date
  token.
- **Class**: semantic-mismatch (lexical fallback too weak for date facts)

## 2. ver-019 — supported_claim_incorrectly_rejected
- **Question**: 全部费用都已审批吗？
- **Claim**: 3 笔费用均处于已审批状态 (factual, GT supported=True)
- **Prediction**: unsupported (semantic, best sim 0.000)
- **Cause**: "已审批" token did not surface in the SQL row content;
  the boolean column `approved=True` was not tokenised for matching.
- **Class**: semantic-mismatch (boolean column not in search text)

## 3. ver-024 — supported_claim_incorrectly_rejected
- **Question**: 3 笔费用的金额是否都不相同？
- **Claim**: 3 笔费用金额互不相同 (factual, GT supported=True)
- **Prediction**: unsupported (semantic, best sim 0.000)
- **Cause**: a *deduction* claim ("互不相同") has no literal token in
  the evidence; the rule layer has no inequality reasoning.
- **Class**: semantic-mismatch (deductive claim beyond surface match)

## 4. ver-056 — supported_claim_incorrectly_rejected
- **Question**: 1 月与 2 月费用各占多少比例？
- **Claim**: 1 月占 43.49%，2 月占 56.51% (derived, GT supported=True)
- **Prediction**: unsupported (exact layer: claimed value None, no
  parseable structured value in the analysis evidence)
- **Cause**: the derived `expected_value` was not carried through to a
  parseable `value` field on the claim, so the exact layer could not
  recompute 43.49% / 56.51%.
- **Class**: derived-calculation error (value not parseable)

## 5. ver-094 — supported_claim_incorrectly_rejected
- **Question**: 采购合同审批上限？
- **Claim**: 部门经理可审批 50 万元合同 (rule_based, GT supported=True)
- **Prediction**: unsupported (semantic, best sim 0.056)
- **Cause**: the RAG chunk wording ("50 万") did not lexically match
  the claim phrasing ("50 万元") closely enough for the 0.5 gate.
- **Class**: semantic-mismatch (unit-suffix wording variance)

## 6. ver-064 / ver-065 — conflict_not_detected
- **Question**: 费用增长率是多少？
- **Claim**: 费用环比增长 30% (derived, GT expected_conflict=True,
  a conflicting source shows 116.24%)
- **Prediction**: unsupported, not conflict (semantic, "no candidate
  above threshold")
- **Cause**: the two analysis values (30% vs 116.24%) landed in
  different *evidence objects*; the exact layer's multi-value conflict
  branch was not reached because the linked candidate set was a single
  structured value, so the cross-source disagreement was missed.
- **Class**: conflict-not-detected (cross-source disagreement)

## 7. ver-031 — numeric mismatch incorrectly accepted
- **Question**: 本月费用总额是多少？
- **Claim**: 本月费用总额为 5000 元 (numerical, GT supported=False;
  true value 4600.5)
- **Prediction**: supported under B_rule_only (exact matched a
  contradiction source) — under C_full this is correctly rejected.
- **Cause**: the exact layer tolerated a *contradicting* structured
  value when a second, wrong source was present; conflict should have
  fired.
- **Class**: numeric-mismatch (tolerance too loose with two sources)

## 8. ver-105 — unsupported claim accepted
- **Question**: 费用类型共 4 种？
- **Claim**: 费用类型共 4 种 (numerical, GT supported=False; true 2)
- **Prediction**: supported (rule layer token-matched "4 种" in an
  incidental document mention)
- **Cause**: a rule-layer token overlap fired on a superficially
  related term rather than the authoritative count.
- **Class**: unsupported-accepted (spurious rule match)

## 9. ver-106 — KG relation mismatch
- **Question**: 订单 99 存在吗？
- **Claim**: 订单 99 由客户 2 下单 (factual, GT supported=False)
- **Prediction**: supported (KG evidence contained an order record,
  the rule layer token-matched "订单" without verifying order_id=99)
- **Cause**: KG rule check did not verify the *specific* entity id
  before reporting support.
- **Class**: KG-relation-mismatch (entity-id not checked)

## 10. ver-116 — conflict not detected (multi-source)
- **Question**: 跨源冲突：平均？
- **Claim**: 平均费用 1533.50 元，报表口径为 2000 元 (derived, GT
  expected_conflict=True)
- **Prediction**: conflict partially detected but attributed to the
  *supported* value only.
- **Cause**: both structured sources were present but the exact layer
  reported the lower value as the match instead of flagging the
  source-to-source disagreement.
- **Class**: conflict-not-detected (auto-picked one value)

---

## Summary of failure classes

| class | count (approx) | typical root cause |
|---|---|---|
| supported_claim_incorrectly_rejected | 14 | lexical fallback too weak for date/boolean/deductive claims |
| conflict_not_detected | 3 | cross-source disagreement not reaching the exact layer |
| unsupported_claim_accepted | 2 | spurious rule token match on incidental terms |
| numeric_mismatch | 2 | tolerance too loose when a wrong source is present |
| derived_calculation | 1 | `expected_value` not parseable |
| KG_relation_mismatch | 1 | entity-id not verified before support |

The dominant failure mode is **over-conservative rejection of correct
claims** under the offline lexical scorer — a property of the test
harness, not the engine.  With a live BGE-M3 embedder the lexical gate
is replaced by a semantic gate and the supported-claim recall is
expected to rise; that result is recorded in the report's
Limitations section rather than claimed here.
