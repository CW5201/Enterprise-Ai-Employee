#!/usr/bin/env python3
"""Phase 2.1 环境检查脚本（不写任何功能代码，只验证真实环境）。

检查项：
  1. EMBEDDING_MODEL 是否可用（本地目录存在）
  2. BGE-M3 是否能真实加载
  3. BGE-M3 实际输出 dimension（前几个值 / norm）
  4. Milvus 服务是否可连接（uri / token）
  5. pymilvus 版本
  6. 当前 collection 是否存在 / entity_count

用法（项目根目录）::

    python scripts/check_env.py
"""

from __future__ import annotations

import math
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 让 .env 的变量生效（EMBEDDING_MODEL / MILVUS_HOST 等）
from src.core.llm_client import _load_dotenv  # noqa: E402

_load_dotenv()

import os  # noqa: E402

import yaml  # noqa: E402

from src.core.embedder import BGE_M3_DIM  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z0-9_]+)(?::-(.*?))?\}")


def _line(label: str, ok_flag: bool, detail: str = "") -> None:
    mark = "OK " if ok_flag else "FAIL"
    print(f"[{mark}] {label}{'  ' + detail if detail else ''}")


def _resolve(value: str) -> str:
    """Resolve ${VAR:-default} placeholders against the real environment."""

    def _sub(match: re.Match[str]) -> str:
        found = os.environ.get(match.group(1))
        if found is not None:
            return found
        default = match.group(2)
        return default if default is not None else ""

    return _ENV_VAR_RE.sub(_sub, str(value))


def main() -> int:
    with (PROJECT_ROOT / "config" / "settings.yaml").open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    ok = True
    print("=" * 60)
    print("Phase 2.1 环境检查")
    print("=" * 60)

    # --- 1. embedding config -------------------------------------------------
    embed_cfg = cfg.get("embedding", {})
    model_source = _resolve(str(embed_cfg.get("model_name", ""))) or "BAAI/bge-m3"
    device = _resolve(str(embed_cfg.get("device", "cpu")))
    print(f"EMBEDDING_MODEL = {model_source}  device = {device}")
    model_path_ok = bool(model_source) and Path(model_source).is_dir()
    print(f"[{'OK' if model_path_ok else 'FAIL'}] 1. EMBEDDING_MODEL 可用  exists={Path(model_source).is_dir()}")
    ok = ok and model_path_ok

    # --- 2. BGE-M3 real load -------------------------------------------------
    from src.core.embedder import create_embedder

    t0 = time.time()
    embedder = create_embedder()
    load_time = time.time() - t0
    print(f"embedder.backend = {embedder.backend}  load_time = {load_time:.1f}s")
    if embedder.backend != "bge-m3":
        print(f"[FAIL] 2. BGE-M3 真实加载  status={embedder.status}")
        print()
        print("环境检查存在 FAIL，请先修复环境。")
        return 1
    print(f"[OK] 2. BGE-M3 真实加载  ({load_time:.1f}s)")

    # --- 3. real embedding ----------------------------------------------------
    test_text = "员工差旅费用报销需要哪些审批？"
    vec = embedder(test_text)
    dim = len(vec)
    norm = math.sqrt(sum(v * v for v in vec))
    print(f"  测试文本: {test_text}")
    print(f"  dimension = {dim}   norm = {norm:.6f}")
    print(f"  前5个值 = {[f'{v:.6f}' for v in vec[:5]]}")
    dim_ok = dim > 0 and abs(norm - 1.0) < 1e-3
    print(f"[{'OK' if dim_ok else 'FAIL'}] 3. embedding 真实生成且已归一化")
    ok = ok and dim_ok

    declared_dim = int(embed_cfg.get("dimension", BGE_M3_DIM))
    dim_match = declared_dim == dim
    print(f"[{'OK' if dim_match else 'FAIL'}] 3b. 与 settings.yaml dimension 一致  declared={declared_dim} actual={dim}")
    ok = ok and dim_match

    # --- 4. Milvus ------------------------------------------------------------
    milvus_cfg = cfg.get("milvus", {})
    host = _resolve(str(milvus_cfg.get("host", "localhost")))
    port = int(_resolve(str(milvus_cfg.get("port", "19530"))))
    user = _resolve(str(milvus_cfg.get("user", "")))
    password = _resolve(str(milvus_cfg.get("password", "")))
    token = f"{user}:{password}" if user or password else ""
    collection = _resolve(str(milvus_cfg.get("collection", "enterprise_knowledge")))
    metric_type = str(milvus_cfg.get("metric_type", "IP"))
    index_type = str(milvus_cfg.get("index_type", "HNSW"))
    print(f"\nMILVUS uri = http://{host}:{port}  user={'<set>' if user else '(none)'}  token={'<set>' if token else '(none)'}")
    print(f"collection = {collection}  metric_type = {metric_type}  index_type = {index_type}")

    try:
        import pymilvus
        from pymilvus import MilvusClient

        print(f"pymilvus version = {pymilvus.__version__}")
        _client = MilvusClient(uri=f"http://{host}:{port}", token=token)
        _line("4. Milvus 连接", True, f"uri=http://{host}:{port} (pymilvus {pymilvus.__version__})")
        collections = _client.list_collections()
        print(f"现有 collections = {collections}")
        exists = _client.has_collection(collection)
        print(f"目标 collection '{collection}' exists = {exists}")
        if exists:
            stats = _client.get_collection_statistics(collection)
            print(f"collection statistics = {stats}")
        _line("5. collection 状态", True, f"'{collection}' exists={exists} (Phase 2.1 由 build_kb.py 创建)")
        _line("6. pymilvus 已安装", True, f"version={pymilvus.__version__}")
    except Exception as exc:  # noqa: BLE001
        _line("4. Milvus 连接", False, str(exc))
        ok = False

    print()
    print("=" * 60)
    print("环境检查通过。可以进入 Phase 2.1 实现。" if ok else "环境检查存在 FAIL，请先修复环境，不要进入下一步。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
