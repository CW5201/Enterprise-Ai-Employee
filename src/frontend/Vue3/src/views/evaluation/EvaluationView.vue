<template>
  <div class="page">
    <div class="page-head">
      <div>
        <h1 class="page-title">实验评测</h1>
        <div class="page-sub">Phase 5 基线 / 消融对比（数据来自本地 artifacts）</div>
      </div>
      <el-button size="small" :loading="loading" @click="load">刷新</el-button>
    </div>

    <el-alert
      v-if="!loading && !summary?.available"
      title="评测结果不可用（artifacts 未生成或被 gitignore）"
      type="info"
      :closable="false"
      show-icon
      class="mt-4"
    />

    <template v-if="summary?.available">
      <!-- Baselines -->
      <div class="panel">
        <div class="panel-head">基线系统</div>
        <el-table :data="baselineRows" size="small">
          <el-table-column prop="name" label="系统" width="180" />
          <el-table-column label="任务成功率" width="110">
            <template #default="{ row }">{{ pct(row.m.task_success_rate) }}</template>
          </el-table-column>
          <el-table-column label="路由准确率" width="110">
            <template #default="{ row }">{{ pct(row.m.route_accuracy) }}</template>
          </el-table-column>
          <el-table-column label="答案精确匹配" width="110">
            <template #default="{ row }">{{ pct(row.m.answer_exact) }}</template>
          </el-table-column>
          <el-table-column label="证据归因准确率" width="130">
            <template #default="{ row }">{{ pct(row.m.evidence_attribution_accuracy) }}</template>
          </el-table-column>
          <el-table-column label="错误率" width="100">
            <template #default="{ row }">{{ pct(row.m.error_rate) }}</template>
          </el-table-column>
          <el-table-column label="平均 LLM 调用" width="120">
            <template #default="{ row }">{{ row.m.llm_calls_avg?.toFixed(1) ?? '—' }}</template>
          </el-table-column>
          <el-table-column label="平均耗时 (ms)" min-width="120">
            <template #default="{ row }">{{ row.m.latency_total_ms_mean?.toFixed(0) ?? '—' }}</template>
          </el-table-column>
        </el-table>
      </div>

      <!-- Ablations -->
      <div v-if="ablationRows.length" class="panel">
        <div class="panel-head">消融实验</div>
        <el-table :data="ablationRows" size="small">
          <el-table-column prop="name" label="配置" width="180" />
          <el-table-column label="任务成功率" width="110">
            <template #default="{ row }">{{ pct(row.m.task_success_rate) }}</template>
          </el-table-column>
          <el-table-column label="路由准确率" width="110">
            <template #default="{ row }">{{ pct(row.m.route_accuracy) }}</template>
          </el-table-column>
          <el-table-column label="答案精确匹配" width="110">
            <template #default="{ row }">{{ pct(row.m.answer_exact) }}</template>
          </el-table-column>
          <el-table-column label="证据归因准确率" width="130">
            <template #default="{ row }">{{ pct(row.m.evidence_attribution_accuracy) }}</template>
          </el-table-column>
          <el-table-column label="平均耗时 (ms)" min-width="120">
            <template #default="{ row }">{{ row.m.latency_total_ms_mean?.toFixed(0) ?? '—' }}</template>
          </el-table-column>
        </el-table>
      </div>

      <!-- Baseline comparison chart -->
      <div v-if="baselineRows.length >= 2" class="panel">
        <div class="panel-head">基线对比（任务成功率）</div>
        <div ref="chartRef" class="chart" />
      </div>

      <!-- Verification summary -->
      <div v-if="Object.keys(summary.verification).length" class="panel">
        <div class="panel-head">Phase 4 验证指标</div>
        <div class="verif-grid">
          <div v-for="(entry, sys) in summary.verification" :key="sys" class="verif-card">
            <div class="verif-name">{{ sys }}</div>
            <div class="verif-metric">支持准确率：{{ pct(entry.support_accuracy) }}</div>
            <div class="verif-metric">不支持检测 F1：{{ pct(entry.unsupported_detection_f1) }}</div>
            <div class="verif-metric">冲突检测准确率：{{ pct(entry.conflict_detection_accuracy) }}</div>
            <div class="verif-metric">证据归因：{{ pct(entry.evidence_attribution_accuracy) }}</div>
          </div>
        </div>
      </div>

      <!-- Experiment report link -->
      <div v-if="summary.report_path" class="panel">
        <div class="panel-head">实验报告</div>
        <div class="report-link-row">
          <el-tag size="small" type="info">{{ summary.report_path }}</el-tag>
          <span class="report-note">Markdown 报告由离线评测脚本生成，不在此处渲染全文</span>
        </div>
        <pre v-if="summary.report_excerpt" class="report-excerpt">{{ summary.report_excerpt.slice(0, 1200) }}…</pre>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from 'vue'
import * as echarts from 'echarts'
import { getEvaluationSummary } from '@/api/evaluation'
import type { EvaluationEntry, EvaluationSummary } from '@/types'

const summary = ref<EvaluationSummary | null>(null)
const loading = ref(false)

const chartRef = ref<HTMLDivElement | null>(null)
let chart: echarts.ECharts | null = null

interface MetricRow {
  name: string
  m: EvaluationEntry['metrics']
}

const baselineRows = computed<MetricRow[]>(() => {
  if (!summary.value) return []
  return Object.entries(summary.value.baselines).map(([name, entry]) => ({
    name,
    m: entry.metrics,
  }))
})

const ablationRows = computed<MetricRow[]>(() => {
  if (!summary.value) return []
  return Object.entries(summary.value.ablations).map(([name, entry]) => ({
    name,
    m: entry.metrics,
  }))
})

function pct(v?: number): string {
  if (v == null) return '—'
  return `${(v * 100).toFixed(1)}%`
}

async function load() {
  loading.value = true
  try {
    summary.value = await getEvaluationSummary()
    await nextTick()
    renderChart()
  } finally {
    loading.value = false
  }
}

function renderChart() {
  if (!chartRef.value || !summary.value?.available) return
  const rows = baselineRows.value
  if (!rows.length) return
  if (!chart) {
    chart = echarts.init(chartRef.value)
  }
  chart.setOption({
    grid: { left: 60, right: 16, top: 24, bottom: 60 },
    xAxis: {
      type: 'category',
      data: rows.map((r) => r.name),
      axisLabel: { fontSize: 11, rotate: 20 },
    },
    yAxis: { type: 'value', max: 1, axisLabel: { formatter: '{value}' } },
    tooltip: { trigger: 'axis' },
    series: [
      {
        type: 'bar',
        data: rows.map((r) => r.m.task_success_rate ?? 0),
        barWidth: 28,
        itemStyle: { color: '#2563eb' },
        label: {
          show: true,
          position: 'top',
          formatter: (p: { value: number }) => `${(p.value * 100).toFixed(1)}%`,
        },
      },
    ],
  })
}

onMounted(load)
onBeforeUnmount(() => {
  chart?.dispose()
  chart = null
})
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

.mt-4 {
  margin-top: var(--eae-space-4);
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

.chart {
  height: 240px;
}

.verif-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: var(--eae-space-3);
}

.verif-card {
  border: 1px solid var(--eae-border);
  border-radius: var(--eae-radius-sm);
  padding: var(--eae-space-3);
}

.verif-name {
  font-size: 13px;
  font-weight: 600;
  margin-bottom: var(--eae-space-2);
  color: var(--eae-text-primary);
}

.verif-metric {
  font-size: 12px;
  color: var(--eae-text-secondary);
  line-height: 1.8;
  font-variant-numeric: tabular-nums;
}

.report-link-row {
  display: flex;
  align-items: center;
  gap: var(--eae-space-2);
  margin-bottom: var(--eae-space-2);
}

.report-note {
  font-size: 12px;
  color: var(--eae-text-muted);
}

.report-excerpt {
  background: var(--eae-surface-alt);
  border: 1px solid var(--eae-border);
  border-radius: var(--eae-radius-sm);
  padding: var(--eae-space-3);
  font-size: 12px;
  line-height: 1.6;
  white-space: pre-wrap;
  max-height: 300px;
  overflow-y: auto;
  color: var(--eae-text-secondary);
}
</style>
