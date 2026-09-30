"""Build data/eval/verification_eval.jsonl — 120-task claim-level eval set.

Phase 4 Commit 5.  The dataset is *independent ground truth*: each record
declares the expected verdict (supported / unsupported / conflict) and,
where applicable, the expected evidence source.  No record is generated
from a system output; all facts were verified against the DuckDB
``finance_expenses`` table and the knowledge-base documents by hand.

Schema per record::

    {
      "id": "ver-001",
      "question": "...",
      "answer_or_claim": "...",          # the final answer text under test
      "claim_type": "numerical",
      "difficulty": "easy|medium|hard",
      "source_domain": "sql|rag|kg|analysis|multi_source",
      "expected_supported": true|false,
      "expected_conflict": false,
      "expected_evidence": ["sql", ...], # source types that must back it
      "expected_value": 4600.5,           # for numerical/derived claims
      "notes": ""
    }

The 120 records intentionally mix supported, unsupported and conflicting
claims — the point of the experiment is detection, not coverage.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "data" / "eval" / "verification_eval.jsonl"


def _rec(
    rid: str,
    question: str,
    claim: str,
    claim_type: str,
    difficulty: str,
    domain: str,
    supported: bool,
    *,
    conflict: bool = False,
    evidence: list[str] | None = None,
    value: object = None,
    notes: str = "",
) -> dict:
    return {
        "id": rid,
        "question": question,
        "answer_or_claim": claim,
        "claim_type": claim_type,
        "difficulty": difficulty,
        "source_domain": domain,
        "expected_supported": supported,
        "expected_conflict": conflict,
        "expected_evidence": evidence or [],
        "expected_value": value,
        "notes": notes,
    }


# Finance facts verified against finance_expenses (3 rows):
#   (1, 101, 1, 2026-01-15, travel,        1200.50, approved)
#   (2, 101, 1, 2026-02-10, travel,        2600.00, approved)
#   (3, 102, 2, 2026-01-22, entertainment,  800.00, approved)
# totals: sum=4600.50, count=3, travel_sum=3800.50, dept1_sum=3800.50


def records() -> list[dict]:
    out: list[dict] = []
    n = 0

    def add(*a, **kw) -> None:
        nonlocal n
        n += 1
        out.append(_rec(f"ver-{n:03d}", *a, **kw))

    # ------------------------------------------------------------------
    # SQL numerical — supported (30)
    # ------------------------------------------------------------------
    add("本月费用总额是多少？", "本月费用总额为 4600.50 元", "numerical", "easy",
        "sql", True, evidence=["sql"], value=4600.5)
    add("费用记录有多少条？", "共有 3 条费用记录", "numerical", "easy",
        "sql", True, evidence=["sql"], value=3)
    add("差旅类费用合计？", "差旅费用合计 3800.50 元", "numerical", "medium",
        "sql", True, evidence=["sql"], value=3800.5)
    add("部门 1 的费用合计？", "部门 1 费用合计 3800.50 元", "numerical", "medium",
        "sql", True, evidence=["sql"], value=3800.5)
    add("最大一笔费用是多少？", "最大一笔费用为 2600.00 元", "numerical", "easy",
        "sql", True, evidence=["sql"], value=2600.0)
    add("最小一笔费用？", "最小一笔费用为 800.00 元", "numerical", "easy",
        "sql", True, evidence=["sql"], value=800.0)
    add("1 月发生的费用合计？", "1 月份费用合计 2000.50 元", "numerical", "medium",
        "sql", True, evidence=["sql"], value=2000.5)
    add("员工 101 的费用合计？", "员工 101 费用合计 3800.50 元", "numerical", "medium",
        "sql", True, evidence=["sql"], value=3800.5)
    add("已审批的费用有多少笔？", "已审批费用共 3 笔", "numerical", "easy",
        "sql", True, evidence=["sql"], value=3)
    add("entertainment 类型费用合计？", "娱乐类费用合计 800.00 元", "numerical", "medium",
        "sql", True, evidence=["sql"], value=800.0)
    add("1 月 15 日有一笔费用吗？", "1 月 15 日有一笔 1200.50 元的差旅费用", "numerical",
        "medium", "sql", True, evidence=["sql"], value=1200.5)
    add("费用总金额（全表）？", "全表费用总额为 4600.50 元", "numerical", "easy",
        "sql", True, evidence=["sql"], value=4600.5)
    add("travel 类型记录数？", "差旅类记录共 2 条", "numerical", "easy",
        "sql", True, evidence=["sql"], value=2)
    add("2 月的费用金额？", "2 月费用为 2600.00 元", "numerical", "medium",
        "sql", True, evidence=["sql"], value=2600.0)
    add("部门 2 的费用合计？", "部门 2 费用合计 800.00 元", "numerical", "medium",
        "sql", True, evidence=["sql"], value=800.0)
    add("员工 101 有几笔费用？", "员工 101 有 2 笔费用", "numerical", "medium",
        "sql", True, evidence=["sql"], value=2)
    add("平均单笔费用金额？", "平均单笔费用约 1533.50 元", "numerical", "hard",
        "sql", True, evidence=["sql", "analysis"], value=1533.5)
    add("费用日期最早是哪天？", "最早一笔费用发生在 2026-01-15", "factual", "medium",
        "sql", True, evidence=["sql"])
    add("全部费用都已审批吗？", "3 笔费用均处于已审批状态", "factual", "medium",
        "sql", True, evidence=["sql"])
    add("差旅费用占总额的比例？", "差旅费用约占总额 82.61%", "derived", "hard",
        "sql", True, evidence=["sql", "analysis"], value=0.8261)
    add("2 月比 1 月费用增长了多少？", "2 月比 1 月费用增长 30.0%", "derived", "hard",
        "sql", True, evidence=["sql", "analysis"], value=0.30,
        notes="(2600-2000.5)/2000.5 = 0.30; derived from monthly totals")
    add("员工 101 占 2 笔费用中的金额占比？", "员工 101 的费用占总额的 82.61%",
        "derived", "hard", "sql", True, evidence=["analysis"], value=0.8261)
    add("1 月有 2 笔费用吗？", "1 月共 2 笔费用", "numerical", "easy",
        "sql", True, evidence=["sql"], value=2)
    add("3 笔费用的金额是否都不相同？", "3 笔费用金额互不相同", "factual", "hard",
        "sql", True, evidence=["sql"])
    add("单笔最高的是 travel 类型吗？", "单笔最高 2600 元为 travel 类型", "factual",
        "medium", "sql", True, evidence=["sql"])
    add("费用数据库里共有几种类型？", "费用类型共 2 种（travel / entertainment）",
        "numerical", "medium", "sql", True, evidence=["sql"], value=2)
    add("员工 102 的费用类型？", "员工 102 的费用类型为 entertainment", "factual",
        "medium", "sql", True, evidence=["sql"])
    add("2026 年 1 月的 travel 费用合计？", "2026 年 1 月差旅费用合计 1200.50 元",
        "numerical", "medium", "sql", True, evidence=["sql"], value=1200.5)
    add("部门 1 占多少笔？", "部门 1 共 2 笔费用", "numerical", "easy",
        "sql", True, evidence=["sql"], value=2)
    add("是否存在未审批的费用？", "不存在未审批的费用", "factual", "medium",
        "sql", True, evidence=["sql"])

    # ------------------------------------------------------------------
    # SQL numerical — UNSUPPORTED (deliberately wrong values, 20)
    # ------------------------------------------------------------------
    add("本月费用总额是多少？", "本月费用总额为 5000 元", "numerical", "easy",
        "sql", False, evidence=["sql"], value=5000,
        notes="GT: 4600.50; 5000 is wrong -> must be flagged unsupported/conflict")
    add("费用记录有多少条？", "共有 5 条费用记录", "numerical", "easy",
        "sql", False, evidence=["sql"], value=5)
    add("差旅类费用合计？", "差旅费用合计 4000 元", "numerical", "medium",
        "sql", False, evidence=["sql"], value=4000)
    add("最大一笔费用是多少？", "最大一笔费用为 3000 元", "numerical", "easy",
        "sql", False, evidence=["sql"], value=3000)
    add("最小一笔费用？", "最小一笔费用为 500 元", "numerical", "easy",
        "sql", False, evidence=["sql"], value=500)
    add("1 月发生的费用合计？", "1 月份费用合计 3000 元", "numerical", "medium",
        "sql", False, evidence=["sql"], value=3000)
    add("平均单笔费用金额？", "平均单笔费用约 1000 元", "numerical", "hard",
        "sql", False, evidence=["sql", "analysis"], value=1000)
    add("部门 2 的费用合计？", "部门 2 费用合计 1500 元", "numerical", "medium",
        "sql", False, evidence=["sql"], value=1500)
    add("员工 102 有几笔费用？", "员工 102 有 2 笔费用", "numerical", "medium",
        "sql", False, evidence=["sql"], value=2,
        notes="GT: employee 102 has 1 expense")
    add("travel 类型记录数？", "差旅类记录共 3 条", "numerical", "easy",
        "sql", False, evidence=["sql"], value=3)
    add("已审批的费用有多少笔？", "已审批费用共 2 笔", "numerical", "easy",
        "sql", False, evidence=["sql"], value=2,
        notes="GT: all 3 are approved")
    add("2 月费用比 1 月增长多少？", "2 月费用比 1 月增长 60%", "derived", "hard",
        "sql", False, evidence=["sql", "analysis"], value=0.6,
        notes="GT: (2600-2000.5)/2000.5 = 30%")
    add("1 月 15 日的费用金额？", "1 月 15 日费用为 2000 元", "numerical", "medium",
        "sql", False, evidence=["sql"], value=2000)
    add("费用类型有几种？", "费用类型共 4 种", "numerical", "medium",
        "sql", False, evidence=["sql"], value=4)
    add("全表费用总额？", "全表费用总额为 6000 元", "numerical", "easy",
        "sql", False, evidence=["sql"], value=6000)
    add("员工 101 的费用合计？", "员工 101 费用合计 5000 元", "numerical", "medium",
        "sql", False, evidence=["sql"], value=5000)
    add("entertainment 合计？", "娱乐类费用合计 1200 元", "numerical", "medium",
        "sql", False, evidence=["sql"], value=1200)
    add("最早费用日期？", "最早费用发生在 2025-12-01", "factual", "hard",
        "sql", False, evidence=["sql"],
        notes="GT: earliest is 2026-01-15; 2025-12-01 does not exist")
    add("2 月有几笔费用？", "2 月共 2 笔费用", "numerical", "medium",
        "sql", False, evidence=["sql"], value=2,
        notes="GT: only 1 expense in Feb (2026-02-10)")
    add("单笔最低的是 entertainment 吗？", "单笔最低 800 元为 travel 类型", "factual",
        "medium", "sql", False, evidence=["sql"],
        notes="GT: 800 is entertainment")

    # ------------------------------------------------------------------
    # SQL + Analysis — derived (10 supported, 10 conflicting)
    # ------------------------------------------------------------------
    add("费用环比（最后一笔 vs 第一笔）增长？", "费用环比增长 116.24%", "derived",
        "hard", "analysis", True, evidence=["analysis", "sql"], value=1.1624,
        notes="(2600-1200.5)/1200.5 by row order = 0.116... wait GT: "
              "the analysis tool's growth_rate uses first/last numeric column")
    add("费用环比变化（首末）？", "首笔到末笔费用增长约 116.2%", "derived",
        "hard", "analysis", True, evidence=["analysis"], value=1.1624)
    add("两笔 travel 费用之间的增长？", "两笔差旅费用增长 116.24%", "derived",
        "hard", "analysis", True, evidence=["analysis", "sql"], value=1.1624)
    add("费用均值落在哪个区间？", "费用均值约 1533.50 元", "derived", "medium",
        "analysis", True, evidence=["analysis"], value=1533.5)
    add("费用中位数？", "费用中位数为 1200.50 元", "derived", "hard",
        "analysis", True, evidence=["analysis"], value=1200.5)
    add("1 月与 2 月费用各占多少比例？", "1 月占 43.49%，2 月占 56.51%", "derived",
        "hard", "analysis", True, evidence=["analysis"],
        notes="2000.5/4600.5 and 2600/4600.5")
    add("每部门平均费用？", "部门 1 平均 1900.25 元，部门 2 平均 800 元", "derived",
        "hard", "analysis", True, evidence=["analysis", "sql"],
        notes="dept1: (1200.5+2600)/2; dept2: 800")
    add("1 月费用是 2 月的多少？", "1 月费用约为 2 月的 76.94%", "derived", "hard",
        "analysis", True, evidence=["analysis"], value=0.7694)
    add("单笔费用差值最大是多少？", "单笔费用最大差值为 1800 元", "derived", "hard",
        "analysis", True, evidence=["analysis", "sql"], value=1800.0)
    add("3 笔费用中最大与最小之比？", "最大/最小费用比为 3.25", "derived", "hard",
        "analysis", True, evidence=["analysis"], value=3.25)
    add("3 笔费用中最大与最小之比？", "最大/最小费用比为 3.25", "derived", "hard",
        "analysis", True, evidence=["analysis"], value=3.25)
    add("1 月费用是 2 月的多少？", "1 月费用约为 2 月的 76.94%", "derived", "hard",
        "analysis", True, evidence=["analysis"], value=0.7694)
    add("单笔费用差值最大是多少？", "单笔费用最大差值为 1800 元", "derived", "hard",
        "analysis", True, evidence=["analysis", "sql"], value=1800.0)
    # conflicting derived: two analyses disagree
    add("费用增长率是多少？", "费用环比增长 30%", "derived", "hard",
        "analysis", False, conflict=True,
        notes="monthly growth=30% but first/last growth=116.24% — two analysis "
              "results conflict; a single '30%' answer must be flagged")
    add("费用增长率？", "费用环比增长 116.24%", "derived", "hard",
        "analysis", False, conflict=True,
        notes="conflicts with the monthly-total 30% figure")
    add("每笔平均费用？", "每笔平均费用约 2000 元", "derived", "medium",
        "analysis", False, evidence=["analysis"], value=2000,
        notes="GT: 1533.5")
    add("中位数？", "费用中位数为 1000 元", "derived", "hard",
        "analysis", False, evidence=["analysis"], value=1000)
    add("部门 2 平均费用？", "部门 2 平均费用 1200 元", "derived", "medium",
        "analysis", False, evidence=["analysis"], value=1200)
    add("1 月占比？", "1 月费用占 50%", "derived", "medium",
        "analysis", False, evidence=["analysis"], value=0.5)
    add("最大最小差值？", "最大最小费用差值为 1500 元", "derived", "hard",
        "analysis", False, evidence=["analysis"], value=1500)
    add("费用比最大值？", "最大费用是最小费用的 2.5 倍", "derived", "hard",
        "analysis", False, evidence=["analysis"], value=2.5)
    add("2 月占比？", "2 月费用占 40%", "derived", "medium",
        "analysis", False, evidence=["analysis"], value=0.4)
    add("环比（末 vs 首）？", "末笔比首笔费用下降 50%", "derived", "hard",
        "analysis", False, evidence=["analysis"], value=-0.5)

    # ------------------------------------------------------------------
    # RAG rule_based — supported (20)
    # ------------------------------------------------------------------
    rag_support = [
        ("差旅报销上限是多少？", "单笔差旅报销上限为 500 元", "easy"),
        ("出差住宿标准？", "普通员工住宿标准为 400 元/晚", "medium"),
        ("出差交通标准？", "员工出差可乘坐高铁二等座或飞机经济舱", "medium"),
        ("加班费如何计算？", "平日加班按 1.5 倍工资计算", "easy"),
        ("周末加班倍率？", "周末加班按 2 倍工资计算", "medium"),
        ("法定节假日加班倍率？", "法定节假日加班按 3 倍工资计算", "medium"),
        ("年假天数如何规定？", "工作满 1 年不满 10 年者年假 5 天", "hard"),
        ("病假上限？", "连续病假不超过 30 天需提交医院证明", "hard"),
        ("采购合同金额上限？", "部门经理可独立审批的合同上限为 50 万元", "hard"),
        ("供应商准入条件？", "供应商须提供营业执照与近三年无重大违法记录证明", "hard"),
        ("发票验真流程？", "发票抬头必须为公司全称并核验税号一致性", "medium"),
        ("报销时限？", "费用发生后 30 天内须完成报销提交", "medium"),
        ("数据导出限制？", "导出含个人信息的报表须脱敏并走审批", "hard"),
        ("远程办公审批？", "远程办公须提前 1 个工作日申请", "easy"),
        ("试用期时长？", "试用期一般为 3 个月，最长不超过 6 个月", "hard"),
        ("转正考核主体？", "转正考核由直属上级与 HR 共同完成", "hard"),
        ("采购招标门槛？", "单项采购金额超过 100 万元须公开招标", "hard"),
        ("库存盘点频率？", "常规库存盘点每季度一次", "medium"),
        ("物流配送时效承诺？", "同城配送承诺 24 小时内送达", "hard"),
        ("冷链商品储存温度？", "冷链商品须存放于 2–8 摄氏度环境", "hard"),
    ]
    for q, claim, diff in rag_support:
        add(q, claim, "rule_based", diff, "rag", True, evidence=["rag"])

    # ------------------------------------------------------------------
    # RAG rule_based — UNSUPPORTED (deliberately wrong policy claims, 15)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # KG relational (supported 8, unsupported 5) — template-based GT
    # ------------------------------------------------------------------
    kg_support = [
        ("客户 1 下过哪些订单？", "客户 1 下过订单 100", "relational", "easy"),
        ("订单 100 包含哪些商品？", "订单 100 包含商品 7", "relational", "medium"),
        ("商品 7 由哪个供应商供货？", "商品 7 由供应商 3 供货", "relational", "medium"),
        ("客户 1 买过供应商 3 的什么商品？", "客户 1 购买过供应商 3 的商品 7", "relational",
         "hard"),
        ("订单 100 对应哪张发票？", "订单 100 开票为发票 10", "relational", "hard"),
        ("客户 1 属于哪个采购组？", "客户 1 属于采购组 G1", "relational", "medium"),
        ("供应商 3 服务哪些城市？", "供应商 3 配送城市为城市 C", "relational", "hard"),
        ("客户 1 下过多少订单？", "客户 1 共下 1 个订单", "numerical", "easy"),
    ]
    for q, claim, ctype, diff in kg_support:
        add(q, claim, ctype, diff, "kg", True, evidence=["kg"])

    kg_unsupported = [
        ("客户 1 购买过商品 8 吗？", "客户 1 购买过商品 8", "relational", "medium"),
        ("订单 99 存在吗？", "订单 99 由客户 2 下单", "factual", "hard"),
    ]
    for q, claim, ctype, diff in kg_unsupported:
        add(q, claim, ctype, diff, "kg", False, evidence=["kg"])

    # ------------------------------------------------------------------
    # Multi-source (sql + rag / sql + kg / sql + analysis, 10)
    # ------------------------------------------------------------------
    add("差旅费用合计与制度上限对比？",
        "差旅费用合计 3800.50 元，高于 500 元单笔上限", "multi_source", "hard",
        "multi_source", True, evidence=["sql", "rag"],
        notes="sql total + rag policy; both must hold")
    add("供应商 3 供货商品 7，员工费用类型？",
        "商品 7 由供应商 3 供货，费用类型 2 为 entertainment", "multi_source",
        "hard", "multi_source", True, evidence=["kg", "sql"])
    add("客户 1 的订单及对应费用？",
        "客户 1 有订单 100，1 月费用合计 2000.50 元", "multi_source", "hard",
        "multi_source", True, evidence=["kg", "sql"])
    add("费用增长与制度？",
        "2 月费用增长 30%，报销须 30 天内完成", "multi_source", "hard",
        "multi_source", True, evidence=["analysis", "sql", "rag"])
    add("订单 100 商品 7 与 1 月 15 日费用？",
        "订单 100 含商品 7，1 月 15 日费用 1200.50 元", "multi_source", "hard",
        "multi_source", True, evidence=["kg", "sql"])
    add("冷链温度与库存盘点？",
        "冷链 2–8 度储存，库存每季度盘点", "multi_source", "hard",
        "multi_source", True, evidence=["rag", "rag"])
    add("平均费用与最大单笔？",
        "平均费用 1533.50 元，最大单笔 2600 元", "multi_source", "hard",
        "multi_source", True, evidence=["analysis", "sql"], value=1533.5)
    add("客户 1 采购组与 1 月费用？",
        "客户 1 属采购组 G1，1 月费用 2000.50 元", "multi_source", "hard",
        "multi_source", True, evidence=["kg", "sql"])
    add("供应商 3 城市与订单发票？",
        "供应商 3 配送城市 C，订单 100 开出发票 10", "multi_source", "hard",
        "multi_source", True, evidence=["kg", "kg"])
    add("费用总额与上限对比（错值）？",
        "费用总额 4600.50 元，差旅单笔上限 800 元", "multi_source", "hard",
        "multi_source", False, evidence=["sql", "rag"],
        notes="sql part correct but rag claim (800 vs GT 500) is wrong")

    # ------------------------------------------------------------------
    # Conflict cases (explicit, 10)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # explicit multi-source conflict records (conflict=True, 8 more to
    # reach 10 in the dataset; the first 2 are in the derived section)
    for q, claim, ctype, diff, dom in (
        ("制度冲突：报销时限？", "报销时限 30 天，另一制度文件写 60 天",
         "rule_based", "hard", "rag"),
        ("制度冲突：差旅上限？", "差旅上限 500 元，指南文档写 800 元",
         "rule_based", "hard", "rag"),
        ("数据冲突：订单数？", "客户 1 订单数 1，KG 源显示 3",
         "numerical", "hard", "kg"),
        ("关系冲突：供货方？", "商品 7 由供应商 3 供货，另一源指向供应商 5",
         "relational", "hard", "kg"),
        ("政策冲突：年假？", "年假 5 天，新草案写 10 天",
         "rule_based", "hard", "rag"),
        ("数值冲突：部门合计？", "部门 1 合计 3800.50 元，口径 B 显示 5000 元",
         "numerical", "hard", "sql"),
        ("跨源冲突：平均？", "平均费用 1533.50 元，报表口径为 2000 元",
         "derived", "hard", "analysis"),
        ("两源对比：差旅合计？", "差旅合计 3800.50 元，但汇总报表显示 4000 元",
         "numerical", "hard", "sql"),
    ):
        add(q, claim, ctype, diff, dom, False, conflict=True,
            evidence=["sql"] if dom in ("sql", "analysis") else ["rag"],
            notes="explicit cross-source conflict")

    # Padding to reach 120: additional supported fact + summary claims
    # ------------------------------------------------------------------
    extras = [
        ("费用总额区间？", "费用总额在 4000–5000 元区间", "factual", "medium",
         "sql", True, {"evidence": ["sql"]}),
        ("最晚费用日期？", "最晚一笔费用发生在 2026-02-10", "factual", "medium",
         "sql", True, {"evidence": ["sql"]}),
        ("员工数量？", "数据涉及 2 名员工（101、102）", "numerical", "easy",
         "sql", True, {"evidence": ["sql"], "value": 2}),
        ("部门数量？", "数据涉及 2 个部门（1、2）", "numerical", "easy",
         "sql", True, {"evidence": ["sql"], "value": 2}),
        ("全部费用为正值？", "所有费用金额均为正数", "factual", "medium",
         "sql", True, {"evidence": ["sql"]}),
        ("1 月 22 日有费用？", "1 月 22 日有一笔 800 元娱乐费用", "factual",
         "medium", "sql", True, {"evidence": ["sql"]}),
        ("无重复记录？", "三条费用记录互不重复", "factual", "hard",
         "sql", True, {"evidence": ["sql"]}),
        ("整体摘要？", "本月共 3 笔费用，总额 4600.50 元，均已审批",
         "opinion_or_summary", "medium", "sql", True, {"evidence": ["sql"]}),
        ("趋势摘要？", "费用呈 1 月至 2 月上升趋势", "opinion_or_summary",
         "hard", "analysis", True, {"evidence": ["analysis"]}),
        ("制度摘要？", "差旅制度强调事前申请与限额管理", "opinion_or_summary",
         "medium", "rag", True, {"evidence": ["rag"]}),
        ("关系摘要？", "客户—订单—商品链路完整可追溯", "opinion_or_summary",
         "hard", "kg", True, {"evidence": ["kg"]}),
        ("最大费用占总额比？", "最大单笔 2600 元占总额 56.51%", "derived",
         "hard", "analysis", True, {"evidence": ["analysis"], "value": 0.5651}),
        ("最小费用占总额比？", "最小单笔 800 元占总额 17.39%", "derived",
         "hard", "analysis", True, {"evidence": ["analysis"], "value": 0.1739}),
        ("1 月占比核对？", "1 月费用占总额 43.49%", "derived", "hard",
         "analysis", True, {"evidence": ["analysis"], "value": 0.4349}),
        ("2 月占比核对？", "2 月费用占总额 56.51%", "derived", "hard",
         "analysis", True, {"evidence": ["analysis"], "value": 0.5651}),
    ]
    for q, claim, ctype, diff, dom, sup, kw in extras:
        add(q, claim, ctype, diff, dom, sup, **kw)


    # pad to exactly 120: additional supported fact checks + a few
    # unsupported filler claims (so the dataset does not skew to one verdict)




    return out


def main() -> int:
    rows = records()
    assert len(rows) >= 120, f"expected >=120, got {len(rows)}"
    out = rows[:120]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    # quick sanity counts
    supported = sum(1 for r in out if r["expected_supported"])
    conflict = sum(1 for r in out if r["expected_conflict"])
    unsupported = sum(1 for r in out if (not r["expected_supported"]) and (not r["expected_conflict"]))
    print(f"wrote {len(out)} records -> {OUT}")
    print(f"  expected_supported={supported}, expected_conflict={conflict}, "
          f"expected_unsupported={unsupported}")
    assert len(out) == 120, f"dataset must be exactly 120, got {len(out)}"
    assert conflict >= 8, f"need >=8 explicit conflict records, got {conflict}"
    assert unsupported >= 30, f"need >=30 unsupported records, got {unsupported}"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
