<template>
  <div>
    <div class="page-head">
      <div>
        <h1 class="page-title">报告中心</h1>
        <div class="page-sub">
          每次完成的任务都会生成一份执行报告（回答 + 证据来源 + 校验结论）
        </div>
      </div>
      <div class="head-actions">
        <el-button size="small" :loading="loading" @click="load">刷新</el-button>
      </div>
    </div>

    <div v-if="reports.length" class="panel">
      <el-table :data="reports" size="small" @row-click="(r: ReportRecord) => openReport(r)">
        <el-table-column prop="title" label="报告" min-width="220" show-overflow-tooltip />
        <el-table-column prop="request_id" label="请求 ID" width="240" show-overflow-tooltip />
        <el-table-column label="证据" width="70">
          <template #default="{ row }">{{ row.evidence_count }}</template>
        </el-table-column>
        <el-table-column label="校验结论" min-width="170">
          <template #default="{ row }">
            支持 {{ row.supported_claims }} / 不支持 {{ row.unsupported_claims }} / 冲突
            {{ row.conflicts }}
          </template>
        </el-table-column>
        <el-table-column prop="created_at" label="生成时间" width="180" />
        <el-table-column label="操作" width="150">
          <template #default="{ row }">
            <el-button size="small" text @click.stop="openReport(row)">预览</el-button>
            <a class="download-link" :href="reportDownloadUrl(row.id)" :download="fileName(row.id)">
              下载
            </a>
          </template>
        </el-table-column>
      </el-table>
    </div>
    <div v-else-if="loadError" class="empty-card panel-error">{{ loadError }}</div>
    <div v-else-if="!loading" class="empty-card">
      暂无报告。在「AI 任务」页提交一条任务后，这里会出现可预览、可下载的执行报告。
    </div>

    <el-drawer v-model="detailVisible" :title="selected?.title || '报告预览'" size="560px">
      <div v-if="detailLoading" class="panel-empty">加载中…</div>
      <div v-else-if="detailError" class="panel-empty panel-error">{{ detailError }}</div>
      <template v-else-if="detail">
        <div class="report-meta">
          <div class="kv"><span>请求 ID</span><span>{{ detail.request_id }}</span></div>
          <div class="kv"><span>生成时间</span><span>{{ detail.created_at }}</span></div>
          <div class="kv"><span>证据条数</span><span>{{ detail.evidence_count }}</span></div>
          <div class="kv"><span>正文长度</span><span>{{ detail.content_chars }} 字符</span></div>
        </div>

        <pre v-if="detail.markdown" class="report-body">{{ detail.markdown }}</pre>
        <div v-else class="panel-empty">该报告装配失败，正文为空。</div>

        <a
          class="download-link mt-4"
          :href="reportDownloadUrl(detail.id)"
          :download="fileName(detail.id)"
        >
          下载 Markdown 文件
        </a>
      </template>
    </el-drawer>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { getReport, listReports, reportDownloadUrl } from '@/api/reports'
import type { ReportDetail, ReportRecord } from '@/types'

const reports = ref<ReportRecord[]>([])
const loading = ref(false)
const loadError = ref('')

const detailVisible = ref(false)
const detail = ref<ReportDetail | null>(null)
const detailLoading = ref(false)
const detailError = ref('')
const selected = ref<ReportRecord | null>(null)

function fileName(id: string): string {
  return `report-${id.slice(0, 8)}.md`
}

async function load() {
  loading.value = true
  try {
    reports.value = await listReports()
    loadError.value = ''
  } catch (err) {
    // Do not swallow: "no reports yet" and "API unreachable" must look different.
    loadError.value = err instanceof Error ? err.message : String(err)
  } finally {
    loading.value = false
  }
}

async function openReport(row: ReportRecord) {
  selected.value = row
  detail.value = null
  detailError.value = ''
  detailLoading.value = true
  detailVisible.value = true
  try {
    detail.value = await getReport(row.id)
  } catch (err) {
    detailError.value = err instanceof Error ? err.message : String(err)
  } finally {
    detailLoading.value = false
  }
}

onMounted(load)
</script>

<style scoped>
.report-meta {
  display: flex;
  flex-direction: column;
  gap: var(--eae-space-2);
  margin-bottom: var(--eae-space-4);
}

.kv {
  display: flex;
  gap: var(--eae-space-3);
  font-size: 13px;
  border-bottom: 1px solid var(--eae-border);
  padding-bottom: 6px;
}

.kv span:first-child {
  flex: 0 0 90px;
  color: var(--eae-text-secondary);
}

.kv span:last-child {
  word-break: break-all;
}

/* The report body is Markdown shown as-is: no Markdown renderer is pulled in
   for it, the text is displayed verbatim in its source form. */
.report-body {
  background: var(--eae-surface-alt);
  border: 1px solid var(--eae-border);
  border-radius: var(--eae-radius-sm);
  padding: var(--eae-space-3);
  font-size: 12px;
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-word;
  margin: 0;
}

.download-link {
  color: var(--eae-primary);
  font-size: 13px;
  text-decoration: none;
  display: inline-block;
}

.download-link:hover {
  text-decoration: underline;
}
</style>
