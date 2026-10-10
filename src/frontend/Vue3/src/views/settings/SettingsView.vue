<template>
  <div>
    <div class="page-head">
      <div>
        <h1 class="page-title">系统设置</h1>
        <div class="page-sub">当前运行状态（只读，配置由服务端管理）</div>
      </div>
      <div class="head-actions">
        <el-button size="small" :loading="loading" @click="load">刷新</el-button>
      </div>
    </div>

    <div class="panel">
      <div class="panel-head">运行时状态（来自 /health）</div>
      <div v-if="loadError" class="panel-empty panel-error">{{ loadError }}</div>
      <div v-else class="kv-list">
        <div v-for="row in healthRows" :key="row.label" class="kv">
          <span>{{ row.label }}</span>
          <el-tag size="small" :type="statusTagType(row.status)">{{ row.status }}</el-tag>
        </div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-head">构建配置（服务端代码固定，见 src/api/main.py）</div>
      <div class="kv-list">
        <div class="kv"><span>检索模式</span><span>hybrid（BM25 + 向量 + RRF）</span></div>
        <div class="kv"><span>动态路由</span><span>已启用（phase3=True）</span></div>
        <div class="kv"><span>结果验证</span><span>已启用（phase4=True）</span></div>
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
import { computed, onMounted, ref } from 'vue'
import { getHealth } from '@/api/health'
import type { HealthResponse } from '@/types'

// Read-only view of the server's actual runtime state.  The values come from
// GET /health - nothing is hardcoded here, so the page cannot claim Milvus is
// in use while the API is actually answering from the in-memory fallback.
const health = ref<HealthResponse | null>(null)
const loading = ref(false)
const loadError = ref('')

const healthRows = computed(() => {
  const h = health.value
  if (!h) return []
  const milvusDetails = h.milvus_details as { collection_name?: string; entity_count?: number } | undefined
  const milvus = h.milvus ?? '未检测'
  const milvusLabel =
    milvus === 'ok' && milvusDetails?.collection_name
      ? `已连接（${milvusDetails.collection_name}，${milvusDetails.entity_count ?? 0} chunks）`
      : milvus
  return [
    { label: 'LLM', status: h.llm ?? '未检测' },
    { label: 'Embedding', status: h.embedding ?? '未检测' },
    { label: '向量库（RAG 后端）', status: milvusLabel },
    { label: '业务数据库（DuckDB）', status: h.database ?? '未检测' },
  ]
})

async function load() {
  loading.value = true
  try {
    health.value = await getHealth()
    loadError.value = ''
  } catch (err) {
    // Do not swallow: an unreachable API and a degraded component differ.
    loadError.value = err instanceof Error ? err.message : String(err)
  } finally {
    loading.value = false
  }
}

function statusTagType(status: string): 'success' | 'danger' | 'warning' | 'info' {
  const s = status.toLowerCase()
  if (s.startsWith('ok') || s.startsWith('已连接')) return 'success'
  if (s.includes('error') || s.includes('fail')) return 'danger'
  if (s.includes('offline') || s.includes('degraded') || s.includes('not found')) return 'warning'
  return 'info'
}

onMounted(load)
</script>

<style scoped>
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
  flex: 0 0 180px;
}

.desc-text {
  font-size: 13px;
  color: var(--eae-text-secondary);
  line-height: 1.7;
}
</style>
