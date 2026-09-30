"""Ground-truth retrieval experiment set for Phase 2.3 Reranker.

Extends the Phase 2.2 local set (``data/eval/hybrid_eval.jsonl``) to a
larger, more representative stage scale: 56 queries over the expanded
knowledge base (27 documents, 190 chunks after ``scripts/build_kb.py``).

Query types covered (per the 2.3 spec):
exact keyword / synonym / multi-keyword / long question / cross-sentence
semantic / numeric-rule / multi-entity / confusable (near-duplicate rules).

``expected_chunk_ids`` were **manually verified** against the chunk
corpus produced by the current chunker (512/64) — see the chunk dump
regenerated from ``src/tools/rag_tool.iter_kb_chunks_for_store``.  They
were NOT derived from any Dense / BM25 / Hybrid result, so ground truth
is independent of the systems under test.

The file also carries an explicit ``difficulty`` label (easy | medium |
hard) used for sliced reporting, and each row keeps its
``source`` (document reference list) and ``category``.

Run ``python scripts/build_kb.py`` first so the chunk corpus is current;
chunk ids are stable across rebuilds of the same corpus.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Ground-truth queries.  ``expected`` lists the chunk ids (kb-xxxxxxxxxx)
# that a human reviewer verified as the correct recall target.  ``src``
# records the source document reference(s) for traceability.
# ---------------------------------------------------------------------------

_Q: list[dict] = [
    # 1. exact keyword — reimbursement
    {"id": "rq-01", "type": "exact_keyword", "difficulty": "easy",
         "query": "报销单",
         "expected": ["kb-128896a6cd", "kb-c9c77d1b7b", "kb-bf2a852433", "kb-111c75cd52"],
         "src": ["finance/finance-0001-travel-reimbursement-policy",
              "finance/finance-0002-travel-reimbursement-guide"], "category": "finance"},
    # 2. numeric / rule — lodging cap
    {"id": "rq-02", "type": "numeric_rule", "difficulty": "easy",
         "query": "住宿标准多少？",
         "expected": ["kb-37202daf2b", "kb-c9c77d1b7b"],
         "src": ["finance/finance-0001-travel-reimbursement-policy"], "category": "finance"},
    # 3. long natural-language question
    {"id": "rq-03", "type": "long_question", "difficulty": "medium",
         "query": "员工出差结束后，应该在多长时间内提交报销单，需要附带哪些材料？",
         "expected": ["kb-128896a6cd", "kb-c9c77d1b7b", "kb-bf2a852433"],
         "src": ["finance/finance-0001-travel-reimbursement-policy",
              "finance/finance-0002-travel-reimbursement-guide"], "category": "finance"},
    # 4. multi-keyword
    {"id": "rq-04", "type": "multi_keyword", "difficulty": "medium",
         "query": "采购申请 供应商 库存",
         "expected": ["kb-3a3372de02", "kb-7811234316", "kb-22e2f20437",
                    "kb-16343ac477", "kb-e5efc91336", "kb-43d6beb8c3"],
         "src": ["operations/ops-0001-inventory-supply-policy",
              "operations/ops-0002-procurement-policy"], "category": "operations"},
    # 5. synonym — remote work
    {"id": "rq-05", "type": "synonym", "difficulty": "medium",
         "query": "远程办公 居家办公 规定",
         "expected": ["kb-60b6d53099", "kb-bbde2f5165", "kb-ef155b850d"],
         "src": ["hr/hr-0001-remote-work-policy"], "category": "hr"},
    # 6. numeric / rule — approval threshold
    {"id": "rq-06", "type": "numeric_rule", "difficulty": "easy",
         "query": "单笔报销超过多少金额需要总经理审批？",
         "expected": ["kb-128896a6cd"],
         "src": ["finance/finance-0001-travel-reimbursement-policy"], "category": "finance"},
    # 7. cross-sentence semantic
    {"id": "rq-07", "type": "cross_sentence", "difficulty": "hard",
         "query": "出差住酒店的标准和报销审批流程是什么？",
         "expected": ["kb-37202daf2b", "kb-128896a6cd", "kb-c9c77d1b7b"],
         "src": ["finance/finance-0001-travel-reimbursement-policy"], "category": "finance"},
    # 8. exact keyword — security
    {"id": "rq-08", "type": "exact_keyword", "difficulty": "easy",
         "query": "一人一号",
         "expected": ["kb-767d78c92d", "kb-cc7db12243"],
         "src": ["security/sec-0001-information-security-policy"], "category": "security"},
    # 9. multi-keyword / semantic mix
    {"id": "rq-09", "type": "multi_keyword", "difficulty": "medium",
         "query": "数据安全 权限 审计 日志",
         "expected": ["kb-51c6a8b4ce", "kb-767d78c92d", "kb-09d6fe14d9"],
         "src": ["security/sec-0001-information-security-policy"], "category": "security"},
    # 10. long semantic question — order approval
    {"id": "rq-10", "type": "long_question", "difficulty": "medium",
         "query": "客户订单的金额在什么范围内由哪个层级审批？超过限额要怎么处理？",
         "expected": ["kb-02144cc672", "kb-3de06e62a7"],
         "src": ["business/bus-0001-order-management-policy"], "category": "business"},

    # 11-15. new-doc exact keywords
    {"id": "rq-11", "type": "exact_keyword", "difficulty": "easy",
         "query": "年假",
         "expected": ["kb-2e3c3f70c1", "kb-42a0175959"],
         "src": ["hr/hr-0002-leave-management-policy"], "category": "hr"},
    {"id": "rq-12", "type": "exact_keyword", "difficulty": "easy",
         "query": "调休",
         "expected": ["kb-3d0e5bde55", "kb-741e399eca"],
         "src": ["hr/hr-0003-attendance-overtime-policy"], "category": "hr"},
    {"id": "rq-13", "type": "exact_keyword", "difficulty": "easy",
         "query": "晋升",
         "expected": ["kb-0ff413fcef", "kb-94e394d5d6"],
         "src": ["hr/hr-0005-performance-promotion-policy"], "category": "hr"},
    {"id": "rq-14", "type": "exact_keyword", "difficulty": "easy",
         "query": "发薪日",
         "expected": ["kb-38c493155e"],
         "src": ["hr/hr-0006-compensation-benefits-policy"], "category": "hr"},
    {"id": "rq-15", "type": "exact_keyword", "difficulty": "easy",
         "query": "三单匹配",
         "expected": ["kb-5661349d7a"],
         "src": ["finance/finance-0004-payables-management-policy"], "category": "finance"},

    # 16-20. new-doc numeric / rule
    {"id": "rq-16", "type": "numeric_rule", "difficulty": "medium",
         "query": "逾期 90 天的应付账款要做什么？",
         "expected": ["kb-b75bd63b53"],
         "src": ["finance/finance-0004-payables-management-policy"], "category": "finance"},
    {"id": "rq-17", "type": "numeric_rule", "difficulty": "medium",
         "query": "单月加班超过 36 小时怎么处理？",
         "expected": ["kb-741e399eca"],
         "src": ["hr/hr-0003-attendance-overtime-policy"], "category": "hr"},
    {"id": "rq-18", "type": "numeric_rule", "difficulty": "medium",
         "query": "应收账款逾期多少天会冻结新订单？",
         "expected": ["kb-c8c46f5c15"],
         "src": ["business/bus-0002-crm-customer-management"], "category": "business"},
    {"id": "rq-19", "type": "numeric_rule", "difficulty": "medium",
         "query": "供应商准时交付率低于多少要预警？",
         "expected": ["kb-3c63026e9a", "kb-7c8c2c5e2a"],
         "src": ["operations/ops-0007-supply-chain-risk-policy"], "category": "operations"},
    {"id": "rq-20", "type": "numeric_rule", "difficulty": "medium",
         "query": "纸质发票要在多少天内完成抵扣认证？",
         "expected": ["kb-a06a2a8df9"],
         "src": ["finance/finance-0005-invoice-tax-compliance"], "category": "finance"},

    # 21-25. new-doc long natural-language
    {"id": "rq-21", "type": "long_question", "difficulty": "medium",
         "query": "新入职员工试用期考核不通过可以怎样处理？",
         "expected": ["kb-643680d8e3"],
         "src": ["hr/hr-0004-recruitment-onboarding-policy"], "category": "hr"},
    {"id": "rq-22", "type": "long_question", "difficulty": "medium",
         "query": "合同金额在 500 万元以上时审批链是怎样的？",
         "expected": ["kb-53319f31fd"],
         "src": ["business/bus-0004-contract-invoicing-policy"], "category": "business"},
    {"id": "rq-23", "type": "long_question", "difficulty": "medium",
         "query": "来料检验抽不合格后如何加严处理？",
         "expected": ["kb-f1b1db2267"],
         "src": ["operations/ops-0005-quality-assurance-policy"], "category": "operations"},
    {"id": "rq-24", "type": "long_question", "difficulty": "medium",
         "query": "盘点差异超过 0.5% 需要做什么？",
         "expected": ["kb-6acb63bf8a"],
         "src": ["operations/ops-0003-warehouse-operations-policy"], "category": "operations"},
    {"id": "rq-25", "type": "long_question", "difficulty": "medium",
         "query": "核心物料中断超过 72 小时启动什么预案？",
         "expected": ["kb-01d243f32d"],
         "src": ["operations/ops-0007-supply-chain-risk-policy"], "category": "operations"},

    # 26-30. new-doc multi-keyword
    {"id": "rq-26", "type": "multi_keyword", "difficulty": "medium",
         "query": "叉车 特种设备 点检",
         "expected": ["kb-d007378f9d"],
         "src": ["operations/ops-0006-safety-equipment-policy"], "category": "operations"},
    {"id": "rq-27", "type": "multi_keyword", "difficulty": "medium",
         "query": "客户准入 资信 授信",
         "expected": ["kb-74b55fc762", "kb-c8c46f5c15", "kb-d14ac17808"],
         "src": ["business/bus-0002-crm-customer-management"], "category": "business"},
    {"id": "rq-28", "type": "multi_keyword", "difficulty": "medium",
         "query": "预算额度 拦截 报销",
         "expected": ["kb-f9e893f091"],
         "src": ["finance/finance-0003-budget-management-policy"], "category": "finance"},
    {"id": "rq-29", "type": "multi_keyword", "difficulty": "medium",
         "query": "钓鱼邮件 数据 上报",
         "expected": ["kb-be3247ddbe"],
         "src": ["security/sec-0002-information-security-operations"], "category": "security"},
    {"id": "rq-30", "type": "multi_keyword", "difficulty": "medium",
         "query": "配送 波次 截单",
         "expected": ["kb-1d7a51ac1e"],
         "src": ["operations/ops-0004-logistics-delivery-policy"], "category": "operations"},

    # 31-35. new-doc cross-sentence semantic
    {"id": "rq-31", "type": "cross_sentence", "difficulty": "hard",
         "query": "出差住宿超标了，超额部分能否报销？",
         "expected": ["kb-c9c77d1b7b", "kb-37202daf2b"],
         "src": ["finance/finance-0001-travel-reimbursement-policy"], "category": "finance"},
    {"id": "rq-32", "type": "cross_sentence", "difficulty": "hard",
         "query": "供应商连续两季评分不合格会怎样处理？",
         "expected": ["kb-7811234316"],
         "src": ["operations/ops-0001-inventory-supply-policy"], "category": "operations"},
    {"id": "rq-33", "type": "cross_sentence", "difficulty": "hard",
         "query": "离职员工的账号和权限应在什么时候回收？",
         "expected": ["kb-767d78c92d", "kb-11f7bb5914"],
         "src": ["security/sec-0001-information-security-policy",
              "security/sec-0004-access-control-policy"], "category": "security"},
    {"id": "rq-34", "type": "cross_sentence", "difficulty": "hard",
         "query": "订单取消后已开票要怎么冲销？",
         "expected": ["kb-4bc835ef9f", "kb-3ad01727d4"],
         "src": ["business/bus-0001-order-management-policy",
              "business/bus-0004-contract-invoicing-policy"], "category": "business"},
    {"id": "rq-35", "type": "cross_sentence", "difficulty": "hard",
         "query": "高敏数据导出前需要哪些审批与措施？",
         "expected": ["kb-51c6a8b4ce", "kb-d96806441a", "kb-4fa47dd313"],
         "src": ["security/sec-0001-information-security-policy",
              "security/sec-0003-data-protection-policy"], "category": "security"},

    # 36-40. confusable — numeric / rule pairs
    {"id": "rq-36", "type": "confusable", "difficulty": "hard",
         "query": "招待费超过多少需要部门负责人与财务负责人双签？",
         "expected": ["kb-8772bb375f"],
         "src": ["finance/finance-0001-travel-reimbursement-policy"], "category": "finance"},
    {"id": "rq-37", "type": "confusable", "difficulty": "hard",
         "query": "超过 50 万元的对客合同由谁审批？",
         "expected": ["kb-53319f31fd"],
         "src": ["business/bus-0004-contract-invoicing-policy"], "category": "business"},
    {"id": "rq-38", "type": "confusable", "difficulty": "hard",
         "query": "供应商付款 5 万元以上由谁审批？",
         "expected": ["kb-a610d85997"],
         "src": ["finance/finance-0004-payables-management-policy"], "category": "finance"},
    {"id": "rq-39", "type": "confusable", "difficulty": "hard",
         "query": "单次预算调整超过部门预算 10% 要怎样处理？",
         "expected": ["kb-ad40f31db8"],
         "src": ["finance/finance-0003-budget-management-policy"], "category": "finance"},
    {"id": "rq-40", "type": "confusable", "difficulty": "hard",
         "query": "大额订单 50 万元以上由谁会签？",
         "expected": ["kb-02144cc672"],
         "src": ["business/bus-0001-order-management-policy"], "category": "business"},

    # 41-45. confusable — policy boundary questions
    {"id": "rq-41", "type": "confusable", "difficulty": "hard",
         "query": "试用期员工能不能申请远程办公？",
         "expected": ["kb-60b6d53099"],
         "src": ["hr/hr-0001-remote-work-policy"], "category": "hr"},
    {"id": "rq-42", "type": "confusable", "difficulty": "hard",
         "query": "病假超过 3 天还需要附带什么证明？",
         "expected": ["kb-2e3c3f70c1"],
         "src": ["hr/hr-0002-leave-management-policy"], "category": "hr"},
    {"id": "rq-43", "type": "confusable", "difficulty": "hard",
         "query": "新客户的账期最长能给多久？",
         "expected": ["kb-c8c46f5c15"],
         "src": ["business/bus-0002-crm-customer-management"], "category": "business"},
    {"id": "rq-44", "type": "confusable", "difficulty": "hard",
         "query": "高敏数据访问为什么要季度复核？",
         "expected": ["kb-767d78c92d", "kb-62c5b65da4"],
         "src": ["security/sec-0001-information-security-policy",
              "security/sec-0004-access-control-policy"], "category": "security"},
    {"id": "rq-45", "type": "confusable", "difficulty": "hard",
         "query": "供应商交期达成率低于 90% 要做什么？",
         "expected": ["kb-afa2cf3348"],
         "src": ["operations/ops-0002-procurement-policy"], "category": "operations"},

    # 46-50. synonym — cross-document concept
    {"id": "rq-46", "type": "synonym", "difficulty": "medium",
         "query": "员工请假",
         "expected": ["kb-2e3c3f70c1", "kb-c644fd43ca", "kb-42a0175959"],
         "src": ["hr/hr-0002-leave-management-policy"], "category": "hr"},
    {"id": "rq-47", "type": "synonym", "difficulty": "medium",
         "query": "加班",
         "expected": ["kb-741e399eca", "kb-3d0e5bde55"],
         "src": ["hr/hr-0003-attendance-overtime-policy"], "category": "hr"},
    {"id": "rq-48", "type": "synonym", "difficulty": "medium",
         "query": "绩效考核",
         "expected": ["kb-94e394d5d6", "kb-9ff142803c", "kb-00b7ac12d1"],
         "src": ["hr/hr-0005-performance-promotion-policy"], "category": "hr"},
    {"id": "rq-49", "type": "synonym", "difficulty": "medium",
         "query": "库存盘点",
         "expected": ["kb-6acb63bf8a", "kb-22e2f20437"],
         "src": ["operations/ops-0003-warehouse-operations-policy",
              "operations/ops-0001-inventory-supply-policy"], "category": "operations"},
    {"id": "rq-50", "type": "synonym", "difficulty": "medium",
         "query": "权限回收",
         "expected": ["kb-11f7bb5914", "kb-767d78c92d"],
         "src": ["security/sec-0004-access-control-policy",
              "security/sec-0001-information-security-policy"], "category": "security"},

    # 51-56. multi-entity — cross-section / cross-document
    {"id": "rq-51", "type": "multi_entity", "difficulty": "hard",
         "query": "合同审批链、客户账期与应收账款逾期冻结",
         "expected": ["kb-53319f31fd", "kb-c8c46f5c15"],
         "src": ["business/bus-0004-contract-invoicing-policy",
              "business/bus-0002-crm-customer-management"], "category": "business"},
    {"id": "rq-52", "type": "multi_entity", "difficulty": "hard",
         "query": "供应商评分、交期达成率与高风险画像",
         "expected": ["kb-7811234316", "kb-afa2cf3348", "kb-134823e572"],
         "src": ["operations/ops-0001-inventory-supply-policy",
              "operations/ops-0002-procurement-policy",
              "operations/ops-0007-supply-chain-risk-policy"], "category": "operations"},
    {"id": "rq-53", "type": "multi_entity", "difficulty": "hard",
         "query": "预算调整、额度占用与付款审批",
         "expected": ["kb-ad40f31db8", "kb-f9e893f091", "kb-a610d85997"],
         "src": ["finance/finance-0003-budget-management-policy",
              "finance/finance-0004-payables-management-policy"], "category": "finance"},
    {"id": "rq-54", "type": "multi_entity", "difficulty": "hard",
         "query": "特权账号、堡垒机与高敏访问审计",
         "expected": ["kb-6694736b4d", "kb-c7ad2aac1c"],
         "src": ["security/sec-0004-access-control-policy"], "category": "security"},
    {"id": "rq-55", "type": "multi_entity", "difficulty": "hard",
         "query": "入职培训、账号开通与试用期转正",
         "expected": ["kb-643680d8e3"],
         "src": ["hr/hr-0004-recruitment-onboarding-policy"], "category": "hr"},
    {"id": "rq-56", "type": "multi_entity", "difficulty": "hard",
         "query": "销售预测偏差、S&OP 会议与库存安全水位",
         "expected": ["kb-0a2f18ce1f", "kb-1c6a6ada4e", "kb-d0241af377",
                    "kb-22e2f20437"],
         "src": ["business/bus-0003-sales-forecasting-policy",
              "operations/ops-0001-inventory-supply-policy"],
         "category": "business"},
]

# de-duplicate expected ids (rq-04 listed ops-0001 supplier mgmt twice)
for q in _Q:
    q["expected_chunk_ids"] = sorted(set(q["expected"]))


def write_jsonl(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for q in _Q:
            fh.write(
                json.dumps(
                    {
                        "id": q["id"],
                        "query": q["query"],
                        "type": q["type"],
                        "difficulty": q["difficulty"],
                        "expected_chunk_ids": q["expected_chunk_ids"],
                        "source": q["src"],
                        "category": q["category"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return len(_Q)


def main() -> int:
    out = PROJECT_ROOT / "data" / "eval" / "retrieval_eval.jsonl"
    n = write_jsonl(out)
    print(f"[build-retrieval-eval] wrote {n} queries to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
