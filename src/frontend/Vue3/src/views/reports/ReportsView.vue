<template>
  <div class="page">
    <div class="page-head">
      <div>
        <h1 class="page-title">报告中心</h1>
        <div class="page-sub">已生成报告列表（报告生成模块尚未接入）</div>
      </div>
      <el-button size="small" :loading="loading" @click="load">刷新</el-button>
    </div>

    <div v-if="reports.length" class="panel">
      <el-table :data="reports" size="small">
        <el-table-column prop="id" label="报告 ID" width="200" />
        <el-table-column prop="title" label="标题" min-width="240" />
        <el-table-column prop="created_at" label="生成时间" width="180" />
      </el-table>
    </div>
    <div v-else-if="loadError" class="panel-empty panel-error">{{ loadError }}</div>
    <div v-else-if="!loading" class="panel-empty">
      暂无报告。报告生成链（src/tools/report_tool.py）将在后续阶段接入。
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import client from '@/api/client'

interface ReportRecord {
  id: string
  title?: string
  created_at?: string
}

const reports = ref<ReportRecord[]>([])
const loading = ref(false)
const loadError = ref('')

async function load() {
  loading.value = true
  try {
    reports.value = await client.get<ReportRecord[]>('/api/reports')
    loadError.value = ''
  } catch (err) {
    loadError.value = err instanceof Error ? err.message : String(err)
  } finally {
    loading.value = false
  }
}

onMounted(load)
</script>

<style scoped>
.page-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  margin-bottom: var(--eae-space-4);
}

.page-sub {
  font-size: 13px;
  color: var(--eae-text-secondary);
}

.panel {
  background: var(--eae-surface);
  border: 1px solid var(--eae-border);
  border-radius: var(--eae-radius);
  padding: var(--eae-space-4);
}

.panel-empty {
  background: var(--eae-surface);
  border: 1px solid var(--eae-border);
  border-radius: var(--eae-radius);
  padding: 48px;
  text-align: center;
  color: var(--eae-text-muted);
  font-size: 13px;
}

.panel-error {
  color: var(--eae-danger);
}
</style>
