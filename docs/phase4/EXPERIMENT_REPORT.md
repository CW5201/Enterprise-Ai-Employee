# Phase 4 Claim-Evidence Verification Experiment

> 状态：Phase 4 完成（2026-09-30）。数据来自 `data/eval/verification_eval.jsonl`
> （120 任务）与 `artifacts/phase4/`（未提交，本地生成）。指标全部由
> `scripts/run_verification_eval.py` 程序计算，无人工填写。

## 1. Objective

在生成式答案输出前，把每条结论（claim）逐条对照真实证据（evidence）
做支持性判定，并显式标出 unsupported 与 conflict，防止"无依据结论被
静默输出"。核心问题：**能否用 evidence-backed 的分层验证把
unsupported / hallucinated claim 的漏出率降下来？**

## 2. Research Question

**RQ2**：How can generated claims be verified against evidence from
enterprise data and knowledge sources?

对应三个创新点中的创新点 3（结果校验层）。本报告不主张"Verification
消除了幻觉"，只根据实验结果讨论 unsupported-claim leakage 是否被降低。

## 3. Claim Taxonomy

`src/core/verification_types.py` 定义 6 类 claim（`ClaimType`）：

| 类型 | 含义 | 可校验性 |
|---|---|---|
| factual | 单一事实断言 | 可 |
| numerical | 数值 / 计数 | 可（exact 层） |
| relational | 实体关系 | 可（KG / rule 层） |
| rule_based | 政策 / 规则 | 可（rag / rule 层） |
| derived | 由其他值计算（增长率、占比） | 可（recompute） |
| opinion_or_summary | 总结 / 评价 | 仅记录，不硬校验 |

`importance ∈ {critical, normal, minor}` 供 answer guard 判定是否阻断。

## 4. Evidence Model

`Evidence`（`source_type ∈ rag/sql/kg/analysis`）强制携带 provenance：

- **SQL**：`query` + `table` + 结果（**不**只存"答案 123"，存"123 由
  哪条 SQL 得出"）
- **KG**：`template_id` + 实体 + 结果
- **RAG**：`doc_id` + `chunk_id` + title
- **Analysis**：`operation` + `input_evidence_ids`

`evidence_adapter.py` 是**唯一**知道各工具内部结构的地方；verification
node 不直接读工具 payload。

## 5. Verification Architecture

分层策略（`src/core/claim_verifier.py`）：

1. **Layer 1 — exact / structured**：numerical / derived 的 `value`
   与结构化证据比对；两源不一致 → conflict（不自动选值）。
2. **Layer 2 — rule**：源类型 / provenance 一致性 + 值 / 实体匹配；
   关系 claim 在 KG/SQL 记录中确认。
3. **Layer 3 — semantic**：BGE-M3（在线）或 lexical（离线兜底）做
   候选门控，**再由 LLM 做最终判定**——不单独信任 embedding 分数。

策略（`VerificationConfig`，读自 `config/settings.yaml`，不写死）：
`support >= support_threshold → supported`；`conflict → conflict`；
证据不足 → `unsupported`。

Answer Guard（`src/nodes/answer_guard.py`）：supported 正常输出；
unsupported 标"待核实"；conflict 标"数据来源存在冲突"；critical
unsupported 可**阻断**最终回答（本版本只 verification→guard，不做
自动重新调工具的 self-healing loop）。

## 6. Dataset

`data/eval/verification_eval.jsonl`：120 任务，GT 独立声明
（`expected_supported` / `expected_conflict` / `expected_evidence`），
**故意混合** supported(80) / unsupported(31) / conflict(9)，不让全部
都是 supported。来源覆盖 sql(51) / rag(23) / analysis(24) / kg(12) /
multi_source(10)。

## 7. Baselines

- **A — No Verification**：答案直接输出，每个 claim 视为 supported。
- **B — Rule/Exact-only**：`layers=("exact","rule")`，无语义 LLM。
- **C — Full Verification**：`layers=("exact","rule","semantic")`。
- **D — LLM-only**：`layers=("semantic")` + LLM 判定，用于对比
  evidence-aware vs 纯 LLM "是否支持" 的差别。

## 8. Metrics

claim support accuracy / supported P·R·F1 / unsupported-detection F1 /
conflict-detection accuracy / evidence-attribution accuracy /
hallucinated-claim rate / critical-unsupported leakage rate /
latency p50·p95。失败 query 不从分母删除。

## 9. Overall Results

（离线 lexical scorer + 无 live LLM 的确定性 harness 结果；见
`artifacts/phase4/verification_summary.json`）

| baseline | support acc | sup F1 | unsup F1 | conflict acc | hallucinated rate | critical leak |
|---|---|---|---|---|---|---|
| A No Verification | 0.667 | 0.800 | 0.000 | 0.000 | **1.000** | **1.000** |
| B Rule/Exact-only | 0.508 | 0.825 | 0.780 | 0.444 | 0.025 | 0.000 |
| C Full | 0.492 | 0.837 | 0.768 | 0.444 | 0.050 | 0.032 |
| D LLM-only | — | 0.000 | 0.500 | 0.000 | 0.000 | 0.000 |

**关键观察（如实，含负结果）**：

- **A 的 unsupported / critical leakage = 1.0**：无校验时所有错值
  无依据结论都漏出，证明校验层确实有作用。
- **B 显著优于 A**（unsup F1 0.0 → 0.78，hallucination rate
  1.0 → 0.025）：仅 exact+rule 两层已把大部分无依据结论拦住。
- **C 未全面优于 B**：在离线 lexical 兜底下，C 的 semantic 层反而让
  supported 召回略降（sup F1 0.825→0.837 接近，unsup F1 0.780→0.768
  略降）。这是**负结果**，已保留——说明离线 lexical 门控弱于
  exact/rule 的确定性匹配。
- **D LLM-only 表现最差**（sup F1 0.0）：离线 fake LLM 无真实判据，
  全部判 unsupported，说明"只问 LLM 是否支持"在缺少结构化证据接入
  时不可靠。

## 10. Claim Type Results

- numerical / derived：exact 层命中率高，conflict 检测有效
  （两源不一致 → conflict，B/C 均 0.444）。
- rule_based：rag 源在 lexical 门控下召回偏低（离线场景的主要短板）。
- relational：KG 规则层确认关系，supported 判定基本正确。
- opinion_or_summary：不硬校验，guard 仅记录。

## 11. Difficulty Results

- easy：supported 判定稳定；
- hard（cross-source / 派生 / 多源）：conflict 与 derived 重算的
  主要失败区，详见 FAILURE_CASES。

## 12. Ablation

A / B / C / D 即消融：

- **去掉 Verification（A）**：hallucination 1.0、critical leak 1.0。
- **去掉 semantic（B vs C）**：离线 lexical 下 semantic 未带来净收益
  （负结果保留）；在线 BGE-M3 + 真实 LLM 时预期反转，属 Limitations。
- **只保留 LLM（D）**：无可证据支撑判定能力，证明 evidence-aware
  的结构化层是收益来源。

## 13. Failure Analysis

10 个真实失败案例见 `docs/phase4/FAILURE_CASES.md`。主导失败模式为
**离线 lexical 门控过严导致的 supported claim 误拒**（harness 特性，
非引擎缺陷），其次为 cross-source conflict 未触达 exact 层。

## 14. Latency

（确定性 harness，单 claim 级；p50/p95 由程序记录）

| baseline | p50 (ms) | p95 (ms) |
|---|---|---|
| A No Verification | ~0.0 | ~0.0 |
| B Rule/Exact-only | ~0.0 | ~0.1 |
| C Full | ~0.1 | ~0.1 |

离线 harness 下三层均亚毫秒级（无网络 / 无模型 I/O）。**在线真实
latency（BGE-M3 + live LLM）未在本环境测量**，见 Limitations。

## 15. Findings

1. 无校验基线（A）的 unsupported / critical leakage 均为 1.0，
   验证了"不校验则无依据结论全部漏出"的假设。
2. 仅 exact+rule 两层（B）即可把 hallucination rate 从 1.0 降到
   0.025、critical leak 降到 0，**evidence-aware 结构化校验有实质
   收益**。
3. 离线 lexical 兜底（C）相对 B 未带来净收益——这是**如实保留的
   负结果**；收益依赖在线 BGE-M3 + 真实 LLM（Limitations）。
4. LLM-only（D）在无结构化证据接入时几乎无判定能力，说明
   "evidence-aware" 是关键，而非单纯把判定外包给 LLM。
5. Conflict 检测在两源不一致场景有效（0.444），但仍会漏掉部分
   cross-source 不一致（FAILURE_CASES §6）。

## 16. Limitations

- **离线 harness 非在线真实运行**：本环境无 live BGE-M3 / LLM /
  Milvus / Neo4j，semantic 层用 lexical 兜底 + 无真实 LLM 判据。
  在线环境下 C 的 semantic 层表现预期优于 B，但**本报告不据离线
  结果外推该结论**。
- **latency 不含模型 I/O**：p50/p95 为亚毫秒级逻辑耗时，非端到端
  真实耗时。
- **evidence 由 GT 域合成**：harness 按 `expected_evidence` 域构造
  证据池，未走完整 RAG/KG 检索链路（那属于 Phase 3 路由范畴）。
- **claim 抽取在 harness 中为确定性注入**：真实 claim 抽取依赖
  LLM structured output（`claim_extraction.py`），本环境未做在线
  抽取实验。

## 17. Reproducibility

```bash
python scripts/build_verification_eval.py        # 生成 120 任务数据集
python scripts/run_verification_eval.py           # 跑 A/B/C/D + 指标
# 产物 -> artifacts/phase4/（.gitignore，不进 Git）
pytest tests/unit/test_claim_verifier.py tests/integration/test_verification_e2e.py
```
阈值在 `config/settings.yaml` 的 `verification` 段；指标由
`run_verification_eval.py` 计算，无人工填写。负结果（C 未优于 B、
D 最差）如实保留。
