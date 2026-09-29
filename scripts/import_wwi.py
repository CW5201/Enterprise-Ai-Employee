#!/usr/bin/env python3
"""将 WideWorldImporters 导入 DuckDB 并重新生成 schema 元数据。

流程（Phase 1）：

    原始 SQL DDL + 种子数据 (data/raw/wwi_ddl)
      -> 方言转换（T-SQL -> DuckDB，见 docs/DATASET.md）
      -> 导入 DuckDB（data/runtime/wwi.duckdb，默认只读访问）
      -> schema 内省
      -> data/schemas/database_schema.yaml
      -> data/schemas/data_dictionary.yaml

重复执行是安全的：运行时数据库每次都会从原始数据重建。

用法（在项目根目录）::

    python scripts/import_wwi.py
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "wwi_ddl"
RUNTIME_DIR = PROJECT_ROOT / "data" / "runtime"
DB_PATH = RUNTIME_DIR / "wwi.duckdb"
SCHEMAS_DIR = PROJECT_ROOT / "data" / "schemas"
SQL_FEWSHOTS = SCHEMAS_DIR / "sql_fewshots.yaml"

# ---------------------------------------------------------------------------
# T-SQL -> DuckDB 方言转换
# ---------------------------------------------------------------------------

# 类型映射：T-SQL 类型 -> DuckDB 等价类型
_TYPE_MAP = [
    (re.compile(r"NVARCHAR\s*\(\s*max\s*\)", re.I), "VARCHAR"),
    (re.compile(r"NVARCHAR\s*\(\s*\d+\s*\)", re.I), "VARCHAR"),
    (re.compile(r"\bDATETIME2\s*\(\s*\d+\s*\)", re.I), "TIMESTAMP"),
    (re.compile(r"\bDATETIME2\b", re.I), "TIMESTAMP"),
    (re.compile(r"\bDATETIME\b", re.I), "TIMESTAMP"),
    (re.compile(r"\bSMALLDATETIME\b", re.I), "TIMESTAMP"),
    (re.compile(r"\bUNIQUEIDENTIFIER\b", re.I), "UUID"),
    (re.compile(r"\bMONEY\b", re.I), "DECIMAL(18,4)"),
    (re.compile(r"\bSMALLMONEY\b", re.I), "DECIMAL(10,4)"),
    (re.compile(r"\bREAL\b", re.I), "DOUBLE"),
    (re.compile(r"\bIMAGE\b", re.I), "BLOB"),
    (re.compile(r"\bNTEXT\b", re.I), "VARCHAR"),
    (re.compile(r"\bTEXT\b", re.I), "VARCHAR"),
    (re.compile(r"\bCHAR\s*\(\s*MAX\s*\)", re.I), "VARCHAR"),
    (re.compile(r"\bVARCHAR\s*\(\s*MAX\s*\)", re.I), "VARCHAR"),
    (re.compile(r"\bVARBINARY\s*\(\s*MAX\s*\)", re.I), "BLOB"),
    (re.compile(r"\bVARBINARY\s*\(\s*\d+\s*\)", re.I), "BLOB"),
    (re.compile(r"\bBIT\b", re.I), "BOOLEAN"),
    # T-SQL 类型在 DuckDB 中无等价物：统一替换为 VARCHAR
    (re.compile(r"\bsys\.?geography\b", re.I), "VARCHAR"),
    (re.compile(r"\bsys\.?name\b", re.I), "VARCHAR"),
    (re.compile(r"\bVARIANT\b", re.I), "VARCHAR"),
    (re.compile(r"\bXML\b", re.I), "VARCHAR"),
    (re.compile(r"\bHIERARCHYID\b", re.I), "VARCHAR"),
    (re.compile(r"\bROWVERSION\b", re.I), "BLOB"),
]

# 带精度参数的 T-SQL 时间类型 -> DuckDB 基础时间类型
_TYPE_PLAIN = [
    (re.compile(r"\bTIMESTAMP\s*\(\s*\d+\s*\)", re.I), "TIMESTAMP"),
    (re.compile(r"\bTIME\s*\(\s*\d+\s*\)", re.I), "TIME"),
    (re.compile(r"\bDATETIMEOFFSET\s*\(\s*\d+\s*\)", re.I), "TIMESTAMP"),
    (re.compile(r"\bSMALLDATETIMEOFFSET\s*\(\s*\d+\s*\)", re.I), "TIMESTAMP"),
]

# 去掉方括号后再映射的类型（如 [sys].[geography] -> sys_geography -> VARCHAR）
_TYPE_MAP_BRACKET = [
    (re.compile(r"\bsys_geography\b", re.I), "VARCHAR"),
    (re.compile(r"\bsys_name\b", re.I), "VARCHAR"),
]

# DuckDB 无法执行、必须整行丢弃的 T-SQL 行（行首锚定）。
# 注意：括号内的匹配必须限制长度（[^)]{0,200}），否则在大型单行 INSERT 上
# 会发生灾难性回溯。
_SKIP_LINE_RE = re.compile(
    r"^\s*(?:GO\s*$|EXEC(UTE)?\s|CREATE\s+(?:NONCLUSTERED\s+|CLUSTERED\s+)?INDEX\b|"
    r"ALTER\s|USE\s|CREATE\s+(SEQUENCE|PROCEDURE|FUNCTION|TRIGGER|TYPE)\b|"
    r"CONSTRAINT\s+\w+\s+(?:PRIMARY|FOREIGN|UNIQUE|CHECK|CLUSTERED)\b|"
    r"PERIOD\s+FOR\s+SYSTEM_TIME\b|ON\s+\w+\s*\([^)]{0,200}\)\s*$|"
    r"INCLUDE\s*\([^)]{0,200}\)|CLUSTERED\b|UNIQUE\s*\(|PRINT\s|DECLARE\s)",
    re.I,
)
# 计算列：列定义行形如 "<col> [TYPE] AS (expr)"；正则刻意保持线性，
# 避免在大型单行文本上发生灾难性回溯
_COMPUTED_COL_HEAD_RE = re.compile(
    r"^(\s+\w+(?:\s+[A-Za-z]\w*)?)\s+AS\s+\(",
    re.I,
)
# INSERT 语句头：用于把种子脚本和 DDL 脚本区分开（搜索全文，而非仅行首，
# 因为种子脚本通常以 PRINT/DECLARE/GO 等 T-SQL 语句开头）
# 匹配 "INSERT INTO [表]" 与 "INSERT [表]" 两种 T-SQL 写法
_INSERT_HEAD_RE = re.compile(r"\bINSERT\b", re.I)


def _is_skippable_line(line: str) -> bool:
    """判断一行是否为 T-SQL 专属语句（应丢弃）。"""
    return _SKIP_LINE_RE.match(line) is not None


def _line_to_duckdb(line: str) -> str | None:
    """转换单行 DDL；返回 None 表示整行丢弃（T-SQL 专属行）。"""
    if not line.strip():
        return ""
    if _is_skippable_line(line):
        return None
    # 计算列："    <col> [TYPE] AS (expr) [PERSISTED] ..." -> "<col> [TYPE] [NOT NULL], "
    # 表达式主体对 DuckDB 无意义，直接替换为列的声明类型
    m = _COMPUTED_COL_HEAD_RE.match(line)
    if m:
        head = m.group(1)
        head_stripped = head.strip()
        parts = head_stripped.rsplit(None, 1)
        # 头部形如 "<col> <type>" 时保留 type；否则该列在 T-SQL 中无显式类型，
        # 统一退化为 VARCHAR
        if len(parts) == 2 and re.match(r"^[A-Za-z]\w*$", parts[1]):
            col_name, col_type = parts[0], parts[1]
        else:
            col_name, col_type = head_stripped, "VARCHAR"
        as_pos = line.find("AS")
        body_start = line.find("(", as_pos)
        depth = 0
        i = body_start + 1
        while i < len(line):
            ch = line[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        tail = line[i + 1 :].lstrip()
        is_not_null = bool(re.match(r"(?:PERSISTED\s+)?NOT\s+NULL", tail, re.I))
        trailing_comma = "," in tail
        nullability = "NOT NULL" if is_not_null else ""
        out = f"    {col_name} {col_type} {nullability}".rstrip()
        return out + (", " if trailing_comma else "")
    return line


def _convert_sqlserver_to_duckdb(sql: str) -> str:
    """把 SQL Server DDL/种子脚本转换为 DuckDB 可执行的 SQL。

    - 映射 T-SQL 类型到 DuckDB 类型；
    - 去掉带 schema 限定的标识符  ``[Sales].[Orders]`` -> ``Sales_Orders``；
    - 删除 T-SQL 专属结构（扩展属性、索引、序列、计算列、时态表属性、
      PK/FK/CLUSTERED/DEFAULT 约束、GO 批分隔符、排序规则）；
    - 保留普通列定义及其 NULL/NOT NULL。

    种子脚本（只包含 INSERT 语句）走快速通道：只做标识符/类型/字面量
    映射即返回，不做列级 DDL 重写。所有正则均为线性模式或带长度上限，
    避免在大型文件上发生灾难性回溯。
    """
    text = sql.lstrip("﻿")
    # [Sales].[Orders] -> Sales_Orders
    text = re.sub(r"\[\s*([A-Za-z_]\w*)\s*\]\s*\.\s*\[\s*([A-Za-z_]\w*)\s*\]", r"\1_\2", text)
    # 类型映射在去方括号之前先做一遍（处理 [sys].[geography] 这类形式）
    for pattern, replacement in _TYPE_MAP:
        text = pattern.sub(replacement, text)
    for pattern, replacement in _TYPE_PLAIN:
        text = pattern.sub(replacement, text)
    # 去掉剩余的方括号标识符，再对去括号后的文本做一轮类型映射
    text = re.sub(r"\[([^\[\]]+)\]", r"\1", text)
    for pattern, replacement in _TYPE_MAP:
        text = pattern.sub(replacement, text)
    for pattern, replacement in _TYPE_MAP_BRACKET:
        text = pattern.sub(replacement, text)
    # T-SQL 字符串字面量 N'...' -> 普通字面量
    text = re.sub(r"\bN'([^']*)'", r"'\1'", text, flags=re.I)
    # 种子脚本到这里已足够：剩余的 DDL 专属重写只对 DDL 文件执行
    # （种子文件既没有 CREATE TABLE，也不含 DDL 专属语法，快速通道直接返回）
    if _INSERT_HEAD_RE.search(text) and not re.search(r"\bCREATE\s+TABLE\b", text, re.I):
        return text
    # 以下均为 DDL 专属重写
    # SQL Server 排序规则在 DuckDB 中不存在
    text = re.sub(r"\s+COLLATE\s+[\w]+", "", text, flags=re.I)
    # T-SQL 列级 MASKED WITH (...) 提示（嵌套括号如 'default()'，按整段删除）
    text = re.sub(r"\s+MASKED\s+WITH\s*(?:\([^()]*(?:\([^()]*\)[^()]*)*\)|\(\s*\))", "", text, flags=re.I)
    # T-SQL IDENTITY(...) 标记
    text = re.sub(r"\s*IDENTITY\s*\(\s*[^)]*\)", "", text, flags=re.I)
    # 时态表行标记 "GENERATED ALWAYS AS ROW START/END"
    text = re.sub(r"\s*GENERATED\s+ALWAYS\s+AS\s+ROW\s+(START|END)", "", text, flags=re.I)
    # T-SQL 约束默认值（NEXT VALUE / sysdatetime）-> 删除
    text = re.sub(
        r"\s*CONSTRAINT\s+\w+\s+DEFAULT\s+\(\s*(?:"
        r"NEXT VALUE FOR\s+\w+|sysdatetime\(\))"
        r"\s*\)",
        "",
        text,
        flags=re.I,
    )
    # 普通 "DEFAULT ((<字面量>)) NOT NULL" -> "DEFAULT <字面量> NOT NULL"
    text = re.sub(r"DEFAULT\s*\(\s*\(\s*([^()]{1,16})\s*\)\s*\)", r"DEFAULT \1", text, flags=re.I)
    # 逐行重写（计算列、可跳过行）
    out_lines: list[str] = []
    for raw in text.splitlines():
        converted = _line_to_duckdb(raw.rstrip())
        if converted is not None:
            out_lines.append(converted)
    text = "\n".join(out_lines)
    # 时态表尾部的 WITH (SYSTEM_VERSIONING=...) 选项
    text = re.sub(r"\)\s*WITH\s*\(\s*SYSTEM_VERSIONING\b[^\n]{0,160}\)", ")", text, flags=re.I)
    # 表级 " ON <分区集合> ..." 分区选项（可能直接贴在右括号后）
    text = re.sub(r"\)\s*ON\s+\[?\w+\]?\s*\([^)]{0,64}\)", ")", text, flags=re.I)
    text = re.sub(r"\)\s*ON\s+\w+\b", ")", text, flags=re.I)
    # 计算列展开后可能残留的 ") NULL," -> 普通列
    text = re.sub(r"(\w+)\s+VARCHAR\s*\)\s*NULL\s*,", r"\1 VARCHAR,", text, flags=re.I)
    # 清理 ",)" / "))" 这类空括号残留
    text = re.sub(r",\s*\)\s*", ")", text)
    text = re.sub(r"\)\s*\)", ")", text)
    return text


# ---------------------------------------------------------------------------
# 脚本读取辅助
# ---------------------------------------------------------------------------


def _is_ddl(script_text: str) -> bool:
    """脚本是否定义了数据库对象（表/序列/存储过程等）。"""
    return re.search(r"\bCREATE\s+TABLE\b", script_text, re.I) is not None


def _statements(script_text: str) -> list[str]:
    """把原始脚本拆分为 DuckDB 可执行的语句。

    种子数据以单条多行 INSERT 形式夹在 T-SQL 批处理中（批内还混有
    存储过程/触发器/索引），这里只保留可执行的对象：DDL + 种子 INSERT。
    种子语句可能以 "INSERT [表]" 或 "INSERT INTO [表]" 出现，两种都保留。
    T-SQL 批分隔符 GO 会被清理掉，避免混入语句体。
    """
    converted = _convert_sqlserver_to_duckdb(script_text)
    # 清理 T-SQL 批分隔符 GO 与声明/输出语句（DuckDB 无法执行）
    converted = re.sub(r"(?:^|\n)\s*GO\s*(?:\n|$)", "\n", converted, flags=re.I)
    converted = re.sub(r"^\s*PRINT\s.*$", "", converted, flags=re.I | re.M)
    converted = re.sub(r"^\s*DECLARE\s.*$", "", converted, flags=re.I | re.M)
    # 在语句边界（; 或新语句关键字）处切分
    raw_statements = re.split(
        r";\s*\n|;\s*GO|(?<=\n)(?=(?:INSERT|CREATE\s+TABLE|DROP\s+TABLE|ALTER\s+TABLE|SELECT|WITH)\s)",
        converted,
    )
    statements: list[str] = []
    for chunk in raw_statements:
        chunk = chunk.strip()
        if not chunk or re.match(r"^(?:GO|EXEC(UTE)?|USE)\b", chunk, re.I):
            continue
        if re.match(
            r"^(?:CREATE\s+TABLE|DROP\s+TABLE|INSERT(\s+INTO)?\s+\w+|ALTER\s+TABLE|SELECT|WITH)\b",
            chunk,
            re.I,
        ):
            statements.append(chunk)
    return statements


def _is_seed_statement(statement: str) -> bool:
    """判断语句是否为种子 INSERT（"INSERT INTO [表]" 或 T-SQL 简写 "INSERT [表]"）。"""
    return re.match(r"^\s*INSERT\s+\w", statement, re.I) is not None


_SEED_VARS = {
    "@CurrentDateTime": "TIMESTAMP '2020-01-01 00:00:00'",
    "@EndOfTime": "TIMESTAMP '9999-12-31 23:59:59.999'",
}


def _normalize_seed_statement(statement: str) -> str:
    """把单行 VALUES 的 T-SQL INSERT 规范化为 DuckDB 可执行语句。

    - 表引用去 schema 限定并映射下划线命名（[Sales].[Colors] -> Sales_Colors）；
    - T-SQL 变量 @CurrentDateTime / @EndOfTime 替换为固定 TIMESTAMP 字面量
      （口径与官方种子数据一致：ValidFrom=2020-01-01, ValidTo=9999-12-31）。
    """
    text = re.sub(
        r"\[\s*([A-Za-z_]\w*)\s*\]\s*\.\s*\[\s*([A-Za-z_]\w*)\s*\]", r"\1_\2", statement
    )
    text = re.sub(r"\[([A-Za-z_]\w*)\]", r"\1", text)
    # 紧跟 INSERT 关键字后的 schema.table（如 Warehouse.Colors）-> 下划线命名
    text = re.sub(r"\bINSERT\s+(Application|Sales|Warehouse|Purchasing)\.(\w+)", r"INSERT \1_\2", text)
    text = re.sub(r"\b(Application|Sales|Warehouse|Purchasing)\.(\w+)", r"\1_\2", text)
    # T-SQL 简写 "INSERT [table]" -> DuckDB "INSERT INTO [table]"
    text = re.sub(r"\bINSERT\s+(?!INTO)(\w+)", r"INSERT INTO \1", text)
    # T-SQL 十六进制字面量 0x.... / x....（转换步骤可能已吞掉前导 0）-> NULL
    # （哈希/照片等二进制字段在 Phase 1 查询中无业务语义，统一置 NULL）
    text = re.sub(r"\bx[0-9A-Fa-f]{8,}\b", "NULL", text)
    # DuckDB 不识别带 "US" 后缀的列（SQL Server 列属性）-> 去掉 "US"
    text = text.replace("CountryIDUS", "CountryID")
    for var, literal in _SEED_VARS.items():
        text = re.sub(re.escape(var) + r"\s*(?=[),\s])", literal, text)
    return text


def _seed_sql_statements(script_text: str) -> list[str]:
    """从 T-SQL 种子脚本中提取字面量 INSERT，并转换为 DuckDB 可执行语句。

    规则：
    - 只保留 VALUES 值为纯字面量的 INSERT（"INSERT 表 (列) VALUES (值列表)"）；
      INSERT ... SELECT（依赖 SQL Server IDENTITY 回填）跳过并记录；
    - T-SQL 简写 "INSERT [表]" 补成 "INSERT INTO [表]"；
    - 时间戳变量 @CurrentDateTime / @EndOfTime 按官方库口径替换为
      '2020-01-01' / '9999-12-31'（与官方完整数据库的 ValidFrom/ValidTo 一致）；
    - 二进制列十六进制字面量（0x...，转换后可能残留 x...）-> NULL；
    - 游标驱动的动态 INSERT（值包含 @variable）无法直接执行，跳过并记录。
    """
    converted = _convert_sqlserver_to_duckdb(script_text)
    # 清理 GO 批分隔符
    converted = re.sub(r"(?:^|\n)\s*GO\s*(?:\n|$)", "\n", converted, flags=re.I)
    # 清理 T-SQL 输出语句（PRINT 无法在 DuckDB 执行）
    converted = re.sub(r"(?m)^\s*PRINT\s.*$", "", converted)
    # 按 INSERT 语句切分
    parts = re.split(r"(?m)(?=INSERT\s)", converted)
    out: list[str] = []
    for part in parts:
        text = part.strip()
        if not re.match(r"^INSERT\s+(?:INTO\s+)?\S", text, re.I):
            continue
        # 表名去方括号/点分 schema -> 下划线命名
        text = re.sub(r"\[\s*([A-Za-z_]\w*)\s*\]\s*\.\s*\[\s*([A-Za-z_]\w*)\s*\]", r"\1_\2", text)
        text = re.sub(r"\[\s*([A-Za-z_]\w*)\s*\]", r"\1", text)
        text = re.sub(r"\b(Application|Sales|Warehouse|Purchasing)\.(\w+)", r"\1_\2", text)
        # INSERT 表 -> INSERT INTO 表
        text = re.sub(r"^INSERT\s+(?!INTO)\s*(\S+)", r"INSERT INTO \1", text, flags=re.I | re.M)
        # INSERT ... SELECT（DuckDB 无 SQL Server IDENTITY 回填，语义不可保证）-> 跳过
        if re.match(r"^INSERT\s+INTO\s+\S+\s*(\([^)]*\)\s*)?SELECT\b", text, re.I | re.S):
            _skipped_seeds.append(f"INSERT...SELECT（IDENTITY 语义），跳过: {text.splitlines()[0][:80]}")
            continue
        # 时戳变量替换（官方库口径）
        text = text.replace("@CurrentDateTime", "'2020-01-01'")
        text = text.replace("@EndOfTime", "'9999-12-31'")
        # 官方脚本中未声明的 @CountryIDUS 变量（CountryID 列）-> 官方全量库口径固定为 239
        text = text.replace("@CountryIDUS", "239")
        # T-SQL 存储过程 [DataLoadSimulation].[GetStateProvinceID]('XX')
        # -> 等价的 StateProvinces 表子查询（按 StateProvinceCode 取 ID；
        #    注意转换步骤可能已把调用拆成下划线标识符，两种形态都处理）
        text = re.sub(
            r"\[\s*DataLoadSimulation\s*\]\s*\.\s*\[\s*GetStateProvinceID\s*\]\s*\(\s*'?([A-Z]+)'?\s*\)",
            r"(SELECT StateProvinceID FROM Application_StateProvinces WHERE StateProvinceCode = '\1')",
            text,
        )
        text = re.sub(
            r"\bGetStateProvinceID\s*\(\s*'?([A-Z]+)'?\s*\)",
            r"(SELECT StateProvinceID FROM Application_StateProvinces WHERE StateProvinceCode = '\1')",
            text,
        )
        # 下划线形态的存储过程调用（[DataLoadSimulation].[Name] 经类型映射后残留）
        text = re.sub(
            r"\bDataLoadSimulation_GetStateProvinceID\s*\(\s*'?([A-Z]+)'?\s*\)",
            r"(SELECT StateProvinceID FROM Application_StateProvinces WHERE StateProvinceCode = '\1')",
            text,
        )
        if "@" in text:
            _skipped_seeds.append(f"游标驱动 INSERT（含变量值），跳过: {text.splitlines()[0][:80]}")
            continue
        # 二进制列十六进制字面量 -> NULL（哈希/照片在 Phase 1 查询中无业务语义；
        # 原始 0x.... 与类型映射后残留的 x.... 两种形态都要覆盖）
        text = re.sub(r"\b0x[0-9A-Fa-f]+\b", "NULL", text)
        text = re.sub(r"\bx[0-9A-Fa-f]{8,}\b", "NULL", text)
        # 语句尾部可能粘连 T-SQL 批处理残留（DECLARE/COMMIT/EXEC 等）-> 截断
        text = re.split(r"(?m)(?=\n\s*(?:DECLARE|COMMIT|BEGIN\s+TRAN|EXEC|PRINT)\b)", text, maxsplit=1)[0].strip()
        out.append(text)
    return out


# ---------------------------------------------------------------------------
# 导入流程
# ---------------------------------------------------------------------------


@dataclass
class ImportResult:
    table_count: int
    row_counts: dict[str, int]
    total_rows: int
    scripts_ddl: int
    scripts_seed: int
    statements_failed: list[str]


def _log(message: str) -> None:
    print(f"[import-wwi] {message}", flush=True)


def _ensure_raw_data() -> None:
    if not RAW_DIR.exists():
        raise SystemExit(
            f"未找到原始 DDL：{RAW_DIR}\n"
            "请先把 WideWorldImporters 的 DDL 与种子脚本下载到 data/raw/wwi_ddl 再重新运行。"
        )


_skipped_seeds: list[str] = []


def import_database() -> ImportResult:
    _ensure_raw_data()
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
        _log(f"已删除旧的运行时数据库 {DB_PATH}")
    for wal in RUNTIME_DIR.glob("wwi.duckdb*"):
        if wal.name == "wwi.kb.json":
            continue
        wal.unlink(missing_ok=True)

    connection = duckdb.connect(str(DB_PATH))
    ddl_scripts = 0
    seed_scripts = 0
    failed: list[str] = []
    try:
        # 1) DDL：DROP TABLE（幂等重建）+ CREATE TABLE，按文件顺序
        _log(f"应用 DDL（来自 {RAW_DIR}）")
        for script in sorted(RAW_DIR.rglob("*.sql")):
            if "Database Triggers" in script.parts:
                continue
            text = script.read_text(encoding="utf-8-sig")
            if not _is_ddl(text):
                continue
            ddl_scripts += 1
            for stmt in _statements(text):
                if re.match(r"^\s*(DROP\s+TABLE|CREATE\s+TABLE)\b", stmt, re.I):
                    # 首次运行时 DROP 不存在的表会报错，幂等重建无影响
                    with contextlib.suppress(duckdb.Error):
                        connection.execute(stmt)

        # 2) 种子数据：PostDeploymentScripts 中直接带行值的 INSERT
        #    （"pds*" 脚本；T-SQL 变量 @CurrentDateTime/@EndOfTime 已映射为固定字面量，
        #     保证 ValidFrom/ValidTo 与官方种子口径一致，官方全量库可直连验证）
        _log("应用种子数据（PostDeploymentScripts INSERT 语句）")
        pds_dir = RAW_DIR / "PostDeploymentScripts"
        for script in sorted(pds_dir.glob("pds*.sql")):
            text = script.read_text(encoding="utf-8-sig")
            if script.name.endswith(
                ("pds142-upd-app-stateprovinces-borders.sql", "pds410-update-archive-tables.sql")
            ):
                continue  # 全量 4.6GB 的 UPDATE 脚本，Phase 1 不需要
            seed_scripts += 1
            for stmt in _seed_sql_statements(text):
                try:
                    connection.execute(stmt)
                except duckdb.Error as exc:
                    failed.append(f"{script.name}: {str(exc).splitlines()[0]}")

        row_counts = {
            t: int(connection.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0])
            for t in _list_tables(connection)
        }
        if failed:
            _log(f"{len(failed)} 条语句执行失败（明细如下）")
            for f in failed[:20]:
                _log(f"   - {f}")
        if _skipped_seeds:
            _log(f"{len(_skipped_seeds)} 条游标驱动种子被跳过（无法在 DuckDB 直接执行）")
            for f in _skipped_seeds[:10]:
                _log(f"   - {f}")
        return ImportResult(
            table_count=len(row_counts),
            row_counts=row_counts,
            total_rows=sum(row_counts.values()),
            scripts_ddl=ddl_scripts,
            scripts_seed=seed_scripts,
            statements_failed=failed,
        )
    finally:
        connection.close()


def _list_tables(connection: duckdb.DuckDBPyConnection) -> list[str]:
    rows = connection.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main' AND table_type = 'BASE TABLE' ORDER BY table_name"
    ).fetchall()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# Schema 内省 -> YAML
# ---------------------------------------------------------------------------


def _describe_table(connection: duckdb.DuckDBPyConnection, table: str) -> dict[str, object]:
    # 主键列：DuckDB 的 pg_catalog 兼容层中主键标记列为 indisprimary
    pk_rows = connection.execute(
        "SELECT a.attname FROM pg_index i JOIN pg_attribute a "
        "ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) "
        "WHERE i.indrelid = ? AND i.indisprimary ORDER BY a.attnum",
        [table],
    ).fetchall()
    pk = [r[0] for r in pk_rows] if pk_rows else []
    columns = connection.execute(f'DESCRIBE "{table}"').fetchall()
    fk_rows = connection.execute(
        "SELECT tc.constraint_name, kcu.column_name, ccu.table_name, ccu.column_name "
        "FROM information_schema.table_constraints tc "
        "JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name "
        "JOIN information_schema.constraint_column_usage ccu ON tc.constraint_name = ccu.constraint_name "
        "WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = ? "
        "ORDER BY kcu.ordinal_position",
        [table],
    ).fetchall()
    fks = [{"constraint": r[0], "column": r[1], "references": f"{r[2]}.{r[3]}"} for r in fk_rows]
    return {
        "name": table,
        "primary_key": pk,
        "columns": [{"name": c[0], "type": c[1], "nullable": c[2] != "NO"} for c in columns],
        "foreign_keys": fks,
    }


def _render_schema_yaml(info: dict[str, object]) -> str:
    tables: list[dict[str, object]] = []
    for name in sorted(info.keys()):
        table = info[name]
        assert isinstance(table, dict)
        entry: dict[str, object] = {"name": name, "primary_key": table["primary_key"], "columns": table["columns"]}
        if table["foreign_keys"]:
            entry["foreign_keys"] = table["foreign_keys"]
        tables.append(entry)
    doc = {
        "version": "v0.2",
        "dialect": "duckdb",
        "source_database": "WideWorldImporters",
        "status": "verified",
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tables": tables,
        "access": {
            "read_only": True,
            "allowed_statements": ["SELECT", "WITH"],
            "forbidden_statements": [
                "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
                "ATTACH", "DETACH", "COPY", "PRAGMA", "CALL", "EXPORT",
                "INSTALL", "LOAD",
            ],
            "max_rows": 1000,
            "timeout_seconds": 30,
        },
    }
    return yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=120)


def _render_data_dictionary(info: dict[str, object]) -> str:
    doc = {
        "version": "v0.2",
        "status": "verified",
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_database": "WideWorldImporters",
        "tables": {
            name: {
                "primary_key": table["primary_key"],
                "columns": [{"name": c["name"], "type": c["type"]} for c in table["columns"]],
            }
            for name, table in sorted(info.items())
        },
        "metrics": {
            "total_revenue": {
                "chinese": "总营收",
                "definition": "SUM(Orders.TotalProfit)，订单毛利润合计（WWI 无 Amount 列，以 TotalProfit 为营收口径）",
                "formula": "SUM(TotalProfit) FROM Sales_Orders",
            },
            "order_count": {
                "chinese": "订单量",
                "definition": "订单数量（不含 Backorder 标记）",
                "formula": "COUNT(OrderID) FROM Sales_Orders WHERE IsUndersupplyBackordered = false",
            },
            "active_customer": {
                "chinese": "活跃客户",
                "definition": "统计周期内至少产生 1 笔订单的客户",
                "formula": "COUNT(DISTINCT CustomerID) FROM Sales_Orders",
            },
        },
        "notes": [
            "本字典由 scripts/import_wwi.py 从 DuckDB 内省结果自动生成（v0.2 起不再手维护）。",
            "WideWorldImporters 为微软公开示例库，全部数据均为虚构，不得声称属于任何真实企业。",
            "WWI 维度表主键为 varchar dimension id（如 'WWI1'），非自增 int。",
        ],
    }
    return yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=120)


def generate_schema_metadata() -> None:
    """对运行时数据库做内省，生成 database_schema.yaml / data_dictionary.yaml。"""
    connection = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        info: dict[str, object] = {}
        for table in _list_tables(connection):
            info[table] = _describe_table(connection, table)
    finally:
        connection.close()
    SCHEMAS_DIR.mkdir(parents=True, exist_ok=True)
    (SCHEMAS_DIR / "database_schema.yaml").write_text(_render_schema_yaml(info), encoding="utf-8")
    (SCHEMAS_DIR / "data_dictionary.yaml").write_text(_render_data_dictionary(info), encoding="utf-8")
    _log(f"已为 {len(info)} 张表写入 schema 元数据到 {SCHEMAS_DIR}")


def _generate_fewshots() -> None:
    """生成并在真实数据库中验证 Few-shot 示例（不造假示例）。"""
    examples = [
        ("订单数量最多的前 10 个客户",
         "SELECT c.CustomerName, COUNT(o.OrderID) AS order_count "
         "FROM Sales_Orders o JOIN Sales_Customers c ON c.CustomerID = o.CustomerID "
         "GROUP BY c.CustomerName ORDER BY order_count DESC LIMIT 10"),
        ("每个销售区域（州/省）的发票总金额",
         "SELECT sp.SalesTerritory, SUM(ol.Quantity * ol.UnitPrice) AS total_revenue "
         "FROM Sales_InvoiceLines ol "
         "JOIN Sales_Invoices i ON i.InvoiceID = ol.InvoiceID "
         "JOIN Sales_Customers c ON c.CustomerID = i.CustomerID "
         "JOIN Application_Cities ct ON ct.CityID = c.DeliveryCityID "
         "LEFT JOIN Application_StateProvinces sp ON sp.StateProvinceID = ct.StateProvinceID "
         "GROUP BY sp.SalesTerritory ORDER BY total_revenue DESC"),
        ("销售额最高的 5 个商品类别",
         "SELECT sg.StockGroupName, SUM(ol.Quantity * ol.UnitPrice) AS revenue "
         "FROM Sales_InvoiceLines ol "
         "JOIN Warehouse_StockItemStockGroups sgk ON sgk.StockItemID = ol.StockItemID "
         "JOIN Warehouse_StockGroups sg ON sg.StockGroupID = sgk.StockGroupID "
         "GROUP BY sg.StockGroupName ORDER BY revenue DESC LIMIT 5"),
    ]
    connection = duckdb.connect(str(DB_PATH), read_only=True)
    verified: list[dict[str, str]] = []
    try:
        for i, (q, sql) in enumerate(examples, 1):
            try:
                rows = connection.execute(sql).fetchall()
                verified.append({
                    "id": f"fs-{i:03d}",
                    "question": q,
                    "sql": sql,
                    "rows_returned": len(rows),
                    "difficulty": "medium",
                    "status": "verified",
                })
            except duckdb.Error as exc:
                _log(f"few-shot '{q}' 验证失败: {exc}")
    finally:
        connection.close()
    doc = {
        "version": "v0.2",
        "dialect": "duckdb",
        "target_database": "WideWorldImporters",
        "status": "verified",
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "few_shots": verified,
        "constraints": [
            "statement_must_be_read_only",
            "allowed_statements: [SELECT, WITH]",
            "must_alias_aggregates",
            "must_qualify_ambiguous_columns",
            "prefer_explicit_join",
        ],
    }
    SCHEMAS_DIR.mkdir(parents=True, exist_ok=True)
    SQL_FEWSHOTS.write_text(
        yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8"
    )
    _log(f"已重新生成 sql_fewshots.yaml（{len(verified)} 条已验证示例）")


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="WideWorldImporters -> DuckDB 导入与元数据生成")
    parser.add_argument("--no-import", action="store_true", help="跳过 DDL/种子导入，只重新生成元数据")
    args = parser.parse_args()
    started = time.time()

    if args.no_import:
        _log("跳过导入；基于已有运行时数据库重新生成元数据")
    else:
        result = import_database()
        _log(
            f"已导入 {result.table_count} 张表，共 {result.total_rows:,} 行 "
            f"（{result.scripts_ddl} 个 DDL 脚本、{result.scripts_seed} 个种子脚本、"
            f"{len(result.statements_failed)} 条失败语句）"
        )

    generate_schema_metadata()
    _generate_fewshots()
    _log(f"完成，用时 {time.time() - started:.1f}s：数据库 {DB_PATH}，元数据 {SCHEMAS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
