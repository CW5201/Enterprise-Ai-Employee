<template>
  <div class="page">
    <h1 class="page-title">系统设置</h1>
    <div class="page-sub">当前运行配置（只读展示，配置由服务端管理）</div>

    <div class="panel">
      <div class="panel-head">运行时参数</div>
      <div class="kv-list">
        <div class="kv"><span>RAG 后端</span><span>{{ settings.rag_backend }}</span></div>
        <div class="kv"><span>检索模式</span><span>{{ settings.retrieval_mode }}</span></div>
        <div class="kv"><span>动态路由</span><span>已启用（Phase 3）</span></div>
        <div class="kv"><span>结果验证</span><span>已启用（Phase 4）</span></div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-head">说明</div>
      <div class="desc-text">
        系统参数（模型端点、Milvus / Neo4j 连接、DuckDB 路径）由
        config/settings.yaml 与环境变量管理，不在此页面编辑。
        如需调整，请修改服务端配置后重启 API。
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

// Read-only view of the server-side runtime configuration. Values are
// static here because config editing is a server-side concern; the page
// documents where each setting lives instead of inventing a fake form.
const settings = ref({
  rag_backend: 'milvus（离线时回退内存 fake 存储）',
  retrieval_mode: 'hybrid（BM25 + 向量 + RRF）',
})
</script>

<style scoped>
.page-sub {
  font-size: 13px;
  color: var(--eae-text-secondary);
  margin-bottom: var(--eae-space-4);
}

.panel {
  background: var(--eae-surface);
  border: 1px solid var(--eae-border);
  border-radius: var(--eae-radius);
  padding: var(--eae-space-4);
  margin-bottom: var(--eae-space-4);
}

.panel-head {
  font-size: 14px;
  font-weight: 600;
  margin-bottom: var(--eae-space-3);
}

.kv-list {
  display: flex;
  flex-direction: column;
}

.kv {
  display: flex;
  gap: var(--eae-space-3);
  font-size: 13px;
  padding: 8px 0;
  border-bottom: 1px solid var(--eae-border);
}

.kv:last-child {
  border-bottom: none;
}

.kv span:first-child {
  color: var(--eae-text-secondary);
  flex: 0 0 140px;
}

.desc-text {
  font-size: 13px;
  color: var(--eae-text-secondary);
  line-height: 1.7;
}
</style>
