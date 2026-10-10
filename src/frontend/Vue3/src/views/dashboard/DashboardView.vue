<template>
  <div>
    <div class="page-head">
      <div>
        <h1 class="page-title">工作台</h1>
        <div class="page-sub">系统状态与常用入口</div>
      </div>
    </div>
    <div class="section">
      <div class="section-head">
        <div>
          <div class="section-title">系统状态</div>
          <div class="section-desc">组件健康检查</div>
        </div>
        <el-button size="small" @click="loadHealth" :loading="loading">刷新</el-button>
      </div>

      <el-table :data="rows" v-if="rows.length" style="width: 100%">
        <el-table-column prop="name" label="组件" width="160" />
        <el-table-column label="状态">
          <template #default="{ row }">
            <el-tag :type="statusTagType(row.status)" size="small">{{ row.status }}</el-tag>
          </template>
        </el-table-column>
      </el-table>

      <div v-else-if="error" class="panel-empty">
        {{ error }}
      </div>
      <div v-else-if="!loading" class="panel-empty">暂无数据</div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { getHealth } from '@/api/health'

interface HealthRow {
  name: string
  status: string
}

const rows = ref<HealthRow[]>([])
const loading = ref(false)
const error = ref('')

async function loadHealth() {
  loading.value = true
  error.value = ''
  try {
    const h = await getHealth()
    rows.value = [
      { name: 'LLM', status: h.llm ?? '未检测' },
      { name: 'Embedding', status: h.embedding ?? '未检测' },
      { name: 'Milvus', status: h.milvus ?? '未检测' },
      { name: 'Database', status: h.database ?? '未检测' },
    ]
  } catch (err) {
    error.value = err instanceof Error ? err.message : '加载失败'
  } finally {
    loading.value = false
  }
}

function statusTagType(status: string): 'success' | 'danger' | 'warning' | 'info' {
  const s = status.toLowerCase()
  if (s.includes('ok') || s === 'completed') return 'success'
  if (s.includes('error') || s.includes('fail')) return 'danger'
  if (s.includes('offline') || s.includes('degraded') || s.includes('not found')) return 'warning'
  return 'info'
}

onMounted(loadHealth)
</script>

<style scoped>
.section {
  background: var(--eae-surface);
  border: 1px solid var(--eae-border);
  border-radius: var(--eae-radius);
  padding: var(--eae-space-4);
  margin-top: var(--eae-space-4);
}

.section-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: var(--eae-space-3);
}
</style>
