<template>
  <div class="page">
    <div class="page-head">
      <div>
        <h1 class="page-title">企业知识</h1>
        <div class="page-sub">知识库文档检索（{{ docs.length }} 篇）</div>
      </div>
      <div class="head-actions">
        <el-input
          v-model="query"
          placeholder="按标题 / 文档 ID 检索"
          :prefix-icon="Search"
          class="search-input"
          clearable
          @keyup.enter="load()"
        />
        <el-select v-model="department" placeholder="部门" clearable class="dept-select" @change="load()">
          <el-option v-for="d in departments" :key="d" :label="d" :value="d" />
        </el-select>
        <el-button @click="load()">检索</el-button>
      </div>
    </div>

    <el-table v-if="docs.length" :data="docs" size="small" @row-click="(r: KnowledgeDocument) => openDetail(r)">
      <el-table-column prop="id" label="文档 ID" width="280" />
      <el-table-column prop="title" label="标题" min-width="200" show-overflow-tooltip />
      <el-table-column prop="department" label="部门" width="100" />
      <el-table-column label="来源" min-width="180" show-overflow-tooltip>
        <template #default="{ row }">{{ row.source }}</template>
      </el-table-column>
      <el-table-column label="类型" width="80">
        <template #default="{ row }">
          <el-tag size="small" :type="row.synthetic ? 'info' : 'success'">
            {{ row.synthetic ? '合成' : '真实' }}
          </el-tag>
        </template>
      </el-table-column>
    </el-table>
    <div v-else-if="!loading" class="panel-empty">未检索到文档</div>

    <el-drawer v-model="detailVisible" :title="selected?.title" size="480px">
      <template v-if="selected">
        <div class="detail-meta">
          <div class="kv"><span>文档 ID</span><span>{{ selected.id }}</span></div>
          <div class="kv"><span>部门</span><span>{{ selected.department }}</span></div>
          <div class="kv"><span>来源</span><span>{{ selected.source }}</span></div>
        </div>
        <pre v-if="content" class="doc-content">{{ content }}</pre>
        <div v-else-if="contentError" class="content-error">{{ contentError }}</div>
        <div v-else-if="contentLoading" class="content-error">加载中…</div>
      </template>
    </el-drawer>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { Search } from '@element-plus/icons-vue'
import { listKnowledgeDocuments } from '@/api/knowledge'
import type { KnowledgeDocument } from '@/types'

const docs = ref<KnowledgeDocument[]>([])
const loading = ref(false)
const query = ref('')
const department = ref('')
const departments = ref<string[]>([])

const detailVisible = ref(false)
const selected = ref<KnowledgeDocument | null>(null)
const content = ref('')
const contentLoading = ref(false)
const contentError = ref('')

async function load() {
  loading.value = true
  try {
    docs.value = await listKnowledgeDocuments({
      q: query.value || undefined,
      department: department.value || undefined,
    })
    const depts = new Set(docs.value.map((d) => d.department))
    departments.value = Array.from(depts).sort()
  } finally {
    loading.value = false
  }
}

async function openDetail(doc: KnowledgeDocument) {
  selected.value = doc
  content.value = ''
  contentError.value = ''
  contentLoading.value = true
  detailVisible.value = true
  try {
    const resp = await fetch(
      `/api/knowledge/documents/${doc.id}/content`,
    )
    if (!resp.ok) {
      contentError.value = `内容加载失败（${resp.status}）`
    } else {
      const data = (await resp.json()) as { content?: string }
      content.value = data.content ?? ''
    }
  } catch {
    contentError.value = '内容加载失败'
  } finally {
    contentLoading.value = false
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
  gap: var(--eae-space-3);
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

.dept-select {
  width: 120px;
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

.detail-meta {
  display: flex;
  flex-direction: column;
  gap: var(--eae-space-2);
  margin-bottom: var(--eae-space-4);
}

.kv {
  display: flex;
  gap: var(--eae-space-2);
  font-size: 13px;
}

.kv span:first-child {
  color: var(--eae-text-muted);
  flex: 0 0 70px;
}

.doc-content {
  background: var(--eae-surface-alt);
  border: 1px solid var(--eae-border);
  border-radius: var(--eae-radius-sm);
  padding: var(--eae-space-3);
  font-size: 12px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 60vh;
  overflow-y: auto;
}

.content-error {
  font-size: 13px;
  color: var(--eae-text-muted);
  padding: var(--eae-space-3);
}
</style>
