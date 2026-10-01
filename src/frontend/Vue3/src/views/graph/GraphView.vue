<template>
  <div class="page">
    <div class="page-head">
      <div>
        <h1 class="page-title">知识图谱</h1>
        <div class="page-sub">实体检索与关系查询（Neo4j）</div>
      </div>
      <div class="head-actions">
        <el-input
          v-model="entitySearch"
          placeholder="检索实体名称"
          :prefix-icon="Search"
          class="search-input"
          clearable
          @keyup.enter="loadEntities()"
        />
        <el-button @click="loadEntities()">检索</el-button>
      </div>
    </div>

    <div v-if="neo4jOffline" class="panel-empty panel-empty-wide">
      Neo4j 未连接（未设置 NEO4J_PASSWORD 或服务离线）。
      实体与关系列表为空，图谱视图不可用。
    </div>

    <div class="graph-layout">
      <!-- Left: entity list -->
      <div class="panel">
        <div class="panel-head">实体（{{ entities.length }}）</div>
        <el-table
          v-if="entities.length"
          :data="entities"
          size="small"
          max-height="420"
          @row-click="(r: GraphEntity) => selectEntity(r)"
        >
          <el-table-column prop="name" label="名称" min-width="140" show-overflow-tooltip />
          <el-table-column prop="type" label="类型" width="110" />
        </el-table>
        <div v-else class="panel-empty">无实体</div>

        <div v-if="selected" class="entity-detail">
          <div class="panel-head">实体详情</div>
          <div class="kv-list">
            <div class="kv"><span>名称</span><span>{{ selected.name }}</span></div>
            <div class="kv"><span>类型</span><span>{{ selected.type }}</span></div>
            <div class="kv"><span>ID</span><span>{{ selected.id }}</span></div>
            <div
              v-for="(v, k) in selected.properties"
              :key="String(k)"
              class="kv"
            >
              <span>{{ k }}</span>
              <span class="prop-value">{{ String(v) }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Right: relations + graph preview -->
      <div class="panel">
        <div class="panel-head">关系（{{ relations.length }}）</div>
        <el-table v-if="relations.length" :data="relations" size="small" max-height="380">
          <el-table-column prop="source" label="起点" min-width="120" show-overflow-tooltip />
          <el-table-column prop="type" label="类型" width="120" />
          <el-table-column prop="target" label="终点" min-width="120" show-overflow-tooltip />
        </el-table>
        <div v-else class="panel-empty">无关系</div>

        <div v-if="relations.length" class="graph-preview">
          <div class="panel-head">图谱预览</div>
          <div ref="graphRef" class="graph-canvas" />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, ref } from 'vue'
import * as echarts from 'echarts'
import { Search } from '@element-plus/icons-vue'
import { listGraphEntities, listGraphRelations } from '@/api/graph'
import type { GraphEntity, GraphRelation } from '@/types'

const entities = ref<GraphEntity[]>([])
const relations = ref<GraphRelation[]>([])
const entitySearch = ref('')
const selected = ref<GraphEntity | null>(null)
const loading = ref(false)

const graphRef = ref<HTMLDivElement | null>(null)
let graphChart: echarts.ECharts | null = null

const neo4jOffline = ref(false)

async function loadEntities() {
  loading.value = true
  try {
    const [ents, rels] = await Promise.all([
      listGraphEntities({ search: entitySearch.value || undefined }),
      listGraphRelations({}),
    ])
    entities.value = ents
    relations.value = rels
    // Both endpoints return 200 [] when Neo4j is offline; two empty lists
    // is the signal to show the offline notice instead of pretending.
    neo4jOffline.value = ents.length === 0 && rels.length === 0
  } finally {
    loading.value = false
  }
}

function selectEntity(e: GraphEntity) {
  selected.value = e
  // refresh relations filtered by this entity
  listGraphRelations({ entity: e.name }).then((rs) => {
    relations.value = rs
    nextTick(renderGraph)
  })
}

function renderGraph() {
  if (!graphRef.value || !relations.value.length) return
  if (!graphChart) {
    graphChart = echarts.init(graphRef.value)
  }

  const nodeSet = new Map<string, string>()
  for (const r of relations.value) {
    nodeSet.set(r.source, r.source)
    nodeSet.set(r.target, r.target)
  }
  const nodes = Array.from(nodeSet.keys()).map((name) => ({
    name,
    symbolSize: 14,
    itemStyle: { color: '#2563eb' },
  }))
  const links = relations.value.map((r) => ({
    source: r.source,
    target: r.target,
    label: { show: true, formatter: r.type, fontSize: 10 },
  }))

  graphChart.setOption({
    tooltip: { trigger: 'item' },
    series: [
      {
        type: 'graph',
        layout: 'force',
        data: nodes,
        links,
        roam: true,
        label: { show: true, fontSize: 11 },
        edgesSymbol: ['none', 'arrow'],
        force: { repulsion: 120, edgeLength: 100 },
        lineStyle: { color: '#9ca3af', width: 1 },
      },
    ],
  })
}

onMounted(loadEntities)
onBeforeUnmount(() => {
  graphChart?.dispose()
  graphChart = null
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

.head-actions {
  display: flex;
  gap: var(--eae-space-2);
  flex: 0 0 auto;
}

.search-input {
  width: 220px;
}

.panel-empty {
  padding: 24px;
  color: var(--eae-text-muted);
  font-size: 13px;
  text-align: center;
}

.panel-empty-wide {
  background: var(--eae-surface);
  border: 1px solid var(--eae-border);
  border-radius: var(--eae-radius);
  margin-bottom: var(--eae-space-4);
}

.graph-layout {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--eae-space-4);
}

@media (max-width: 1100px) {
  .graph-layout {
    grid-template-columns: 1fr;
  }
}

.panel {
  background: var(--eae-surface);
  border: 1px solid var(--eae-border);
  border-radius: var(--eae-radius);
  padding: var(--eae-space-4);
}

.panel-head {
  font-size: 14px;
  font-weight: 600;
  margin-bottom: var(--eae-space-3);
}

.entity-detail {
  margin-top: var(--eae-space-4);
  border-top: 1px solid var(--eae-border);
  padding-top: var(--eae-space-3);
}

.kv-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.kv {
  display: flex;
  gap: var(--eae-space-2);
  font-size: 12px;
}

.kv span:first-child {
  color: var(--eae-text-muted);
  flex: 0 0 80px;
}

.prop-value {
  word-break: break-all;
}

.graph-preview {
  margin-top: var(--eae-space-4);
}

.graph-canvas {
  height: 260px;
  width: 100%;
}
</style>
