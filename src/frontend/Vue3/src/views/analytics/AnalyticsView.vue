<template>
  <div>
    <div class="page-head">
      <div>
        <h1 class="page-title">数据分析</h1>
        <div class="page-sub">KPI、销售趋势、供应商排名、订单统计（数据来自 DuckDB）</div>
      </div>
      <el-button size="small" :loading="loading" @click="load">刷新</el-button>
    </div>

    <el-alert
      v-if="error"
      :title="`数据分析加载失败：${error}`"
      type="error"
      :closable="false"
      show-icon
      class="mt-4"
    />
    <el-alert
      v-else-if="!loading && !summary?.available"
      :title="`数据集不可用${summary?.reason ? '：' + summary.reason : ''}`"
      type="info"
      :closable="false"
      show-icon
      class="mt-4"
    />

    <template v-if="summary?.available">
      <!-- KPI row -->
      <div class="kpi-row">
        <div v-for="kpi in summary.kpis" :key="kpi.label" class="kpi">
          <div class="kpi-label">{{ kpi.label }}</div>
          <div class="kpi-value">
            {{ kpi.value }}<span v-if="kpi.unit" class="kpi-unit">{{ kpi.unit }}</span>
          </div>
        </div>
      </div>

      <div class="charts-grid">
        <div class="panel">
          <div class="panel-head">销售趋势（月度）</div>
          <div v-if="trends?.monthly_sales?.length" ref="trendRef" class="chart" />
          <div v-else class="panel-empty">无趋势数据</div>
        </div>

        <div class="panel">
          <div class="panel-head">供应商排名 Top 10</div>
          <div v-if="trends?.top_suppliers?.length" ref="supplierRef" class="chart" />
          <div v-else class="panel-empty">无供应商数据</div>
        </div>

        <div class="panel">
          <div class="panel-head">客户排名 Top 10</div>
          <div v-if="trends?.top_customers?.length" ref="customerRef" class="chart" />
          <div v-else class="panel-empty">无客户数据</div>
        </div>

        <div class="panel">
          <div class="panel-head">订单统计</div>
          <div v-if="trends?.order_stats" class="kv-list">
            <div class="kv"><span>发票总数</span><span>{{ trends.order_stats.total }}</span></div>
            <div class="kv"><span>红字发票</span><span>{{ trends.order_stats.credit_notes }}</span></div>
            <div class="kv"><span>起始日期</span><span>{{ trends.order_stats.first_invoice }}</span></div>
            <div class="kv"><span>截止日期</span><span>{{ trends.order_stats.last_invoice }}</span></div>
          </div>
          <div v-else class="panel-empty">无订单统计</div>
        </div>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, ref } from 'vue'
import * as echarts from 'echarts'
import { getAnalyticsSummary, getAnalyticsTrends } from '@/api/analytics'
import type { AnalyticsSummary, AnalyticsTrends } from '@/types'

const summary = ref<AnalyticsSummary | null>(null)
const trends = ref<AnalyticsTrends | null>(null)
const loading = ref(false)
const error = ref('')

const trendRef = ref<HTMLDivElement | null>(null)
const supplierRef = ref<HTMLDivElement | null>(null)
const customerRef = ref<HTMLDivElement | null>(null)

const charts: echarts.ECharts[] = []

async function load() {
  loading.value = true
  error.value = ''
  try {
    ;[summary.value, trends.value] = await Promise.all([getAnalyticsSummary(), getAnalyticsTrends()])
    await nextTick()
    renderCharts()
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    loading.value = false
  }
}

function makeChart(el: HTMLDivElement | null, option: echarts.EChartsOption) {
  if (!el) return
  const c = echarts.init(el)
  c.setOption(option)
  charts.push(c)
}

function renderCharts() {
  charts.forEach((c) => c.dispose())
  charts.length = 0

  const monthly = trends.value?.monthly_sales ?? []
  const topSuppliers = trends.value?.top_suppliers ?? []
  const topCustomers = trends.value?.top_customers ?? []

  // Monthly sales — line
  makeChart(trendRef.value, {
    grid: { left: 50, right: 16, top: 24, bottom: 28 },
    xAxis: {
      type: 'category',
      data: monthly.map((m) => m.month),
      axisLabel: { fontSize: 11 },
    },
    yAxis: { type: 'value', axisLabel: { fontSize: 11 } },
    tooltip: { trigger: 'axis' },
    series: [
      {
        type: 'line',
        data: monthly.map((m) => m.total),
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 2 },
        itemStyle: { color: '#2563eb' },
      },
    ],
  })

  // Top suppliers — horizontal bar
  makeChart(supplierRef.value, {
    grid: { left: 120, right: 16, top: 8, bottom: 20 },
    xAxis: { type: 'value', axisLabel: { fontSize: 11 } },
    yAxis: {
      type: 'category',
      data: [...topSuppliers].map((s) => s.name).reverse(),
      axisLabel: { fontSize: 11 },
    },
    tooltip: { trigger: 'axis' },
    series: [
      {
        type: 'bar',
        data: [...topSuppliers].map((s) => s.total).reverse(),
        barWidth: 12,
        itemStyle: { color: '#2563eb', borderRadius: [0, 2, 2, 0] },
      },
    ],
  })

  // Top customers — horizontal bar
  makeChart(customerRef.value, {
    grid: { left: 120, right: 16, top: 8, bottom: 20 },
    xAxis: { type: 'value', axisLabel: { fontSize: 11 } },
    yAxis: {
      type: 'category',
      data: [...topCustomers].map((c) => c.name).reverse(),
      axisLabel: { fontSize: 11 },
    },
    tooltip: { trigger: 'axis' },
    series: [
      {
        type: 'bar',
        data: [...topCustomers].map((c) => c.total).reverse(),
        barWidth: 12,
        itemStyle: { color: '#16a34a', borderRadius: [0, 2, 2, 0] },
      },
    ],
  })
}

function onResize() {
  charts.forEach((c) => c.resize())
}

onMounted(() => {
  window.addEventListener('resize', onResize)
  load()
})
onBeforeUnmount(() => {
  window.removeEventListener('resize', onResize)
  charts.forEach((c) => c.dispose())
})
</script>

<style scoped>
.kpi-row {
  display: grid;
  /* min(Npx, 100%): on a viewport too narrow for Npx the track collapses to
     the full row instead of overflowing it (a bare minmax(Npx, 1fr) forces
     the column past the container and adds a horizontal scrollbar). */
  grid-template-columns: repeat(auto-fit, minmax(min(170px, 100%), 1fr));
  gap: var(--eae-space-3);
  margin-bottom: var(--eae-space-4);
}

.kpi {
  background: var(--eae-surface);
  border: 1px solid var(--eae-border);
  border-radius: var(--eae-radius);
  padding: var(--eae-space-3);
}

.kpi-label {
  font-size: 12px;
  color: var(--eae-text-secondary);
  margin-bottom: 4px;
}

.kpi-value {
  font-size: 20px;
  font-weight: 600;
  color: var(--eae-text-primary);
  font-variant-numeric: tabular-nums;
}

.kpi-unit {
  font-size: 12px;
  font-weight: 400;
  color: var(--eae-text-muted);
  margin-left: 4px;
}

.charts-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(360px, 100%), 1fr));
  gap: var(--eae-space-4);
}

/* grid gap already provides the spacing between panes */
.charts-grid > .panel {
  margin-bottom: 0;
}

.chart {
  height: 240px;
}

.kv-list {
  display: flex;
  flex-direction: column;
  gap: var(--eae-space-2);
}

.kv {
  display: flex;
  justify-content: space-between;
  font-size: 13px;
  padding: 6px 0;
  border-bottom: 1px solid var(--eae-border);
}

.kv:last-child {
  border-bottom: none;
}

.kv span:first-child {
  color: var(--eae-text-secondary);
}

.kv span:last-child {
  font-weight: 500;
  font-variant-numeric: tabular-nums;
}
</style>
