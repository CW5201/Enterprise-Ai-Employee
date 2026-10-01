<template>
  <div class="panel">
    <div class="panel-head">
      回答依据
      <template v-if="result">
        <el-tag v-if="summary.total" size="small" type="success" class="head-tag">
          支持 {{ summary.supported_claims ?? 0 }}
        </el-tag>
        <el-tag v-if="(summary.unsupported_claims ?? 0)" size="small" type="warning" class="head-tag">
          证据不足 {{ summary.unsupported_claims }}
        </el-tag>
        <el-tag v-if="(summary.conflicts ?? 0)" size="small" type="danger" class="head-tag">
          冲突 {{ summary.conflicts }}
        </el-tag>
      </template>
    </div>

    <!-- Claims: per-claim verdicts (never lump the whole answer as "verified") -->
    <div v-if="result && claims.length" class="claims">
      <div v-for="c in claims" :key="c.claim_id" :class="['claim', c.status]">
        <span class="claim-mark">
          {{ c.status === 'supported' ? '✓' : c.status === 'conflict' ? '!' : '⚠' }}
        </span>
        <div class="claim-body">
          <div class="claim-status-line">
            <span class="claim-status">{{ statusLabel(c.status) }}</span>
            <span v-if="c.support_score != null" class="claim-score">
              置信度 {{ (c.support_score * 100).toFixed(0) }}%
            </span>
          </div>
          <div v-if="c.reason" class="claim-reason">{{ c.reason }}</div>
          <div v-if="c.evidence_ids?.length" class="claim-refs">
            证据：{{ c.evidence_ids.map((id) => evLabel(id)).join(' · ') }}
          </div>
        </div>
      </div>
    </div>
    <div v-else-if="result" class="panel-empty-sm">无 claim 级验证结果</div>

    <!-- Evidence -->
    <div v-if="result && result.evidence?.length" class="evidence">
      <div class="subhead">证据</div>
      <div v-for="ev in result.evidence" :key="ev.evidence_id" class="ev-row">
        <el-tag size="small" class="ev-type">{{ ev.source_type }}</el-tag>
        <span class="ev-ref">{{ ev.source_ref }}</span>
        <span v-if="ev.score != null" class="ev-score">相似度 {{ (ev.score * 100).toFixed(0) }}%</span>
        <div v-if="evDetail(ev)" class="ev-detail">{{ evDetail(ev) }}</div>
      </div>
    </div>

    <!-- Tool results -->
    <div v-if="result && result.tool_calls?.length" class="toolresults">
      <div class="subhead">工具结果</div>
      <div v-for="(tc, i) in result.tool_calls" :key="i" class="toolrow">
        <el-tag size="small" :type="tc.ok ? 'success' : 'danger'">{{ tc.tool }}</el-tag>
        <span v-if="tc.ok" class="toolok">ok</span>
        <span v-else class="toolfail">{{ tc.error || '执行失败' }}</span>
      </div>
    </div>

    <div v-if="!result" class="panel-empty-sm">执行任务后显示依据</div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { ChatResponse, EvidenceItem, VerificationResult, VerificationSummary } from '@/types'

const props = defineProps<{ result: ChatResponse | null }>()

const summary = computed<VerificationSummary>(
  () => (props.result?.verification_summary ?? {}) as VerificationSummary,
)

const claims = computed<VerificationResult[]>(() => props.result?.verification ?? [])

function statusLabel(s: string): string {
  return s === 'supported' ? '已支持' : s === 'conflict' ? '数据冲突' : '证据不足'
}

function evDetail(ev: EvidenceItem): string {
  const p = ev.payload ?? {}
  switch (ev.source_type) {
    case 'duckdb':
    case 'sql': {
      const cols = (p.columns as string[] | undefined)?.join(' | ')
      const n = p.row_count
      if (cols) return `${cols}${n != null ? `（${n} 行）` : ''}`
      return ev.content?.slice(0, 120) ?? ''
    }
    case 'milvus':
    case 'rag':
      return ev.content?.slice(0, 120) || ev.source_ref
    case 'kg':
      return JSON.stringify(p).slice(0, 120)
    default:
      return ev.content?.slice(0, 120) || ''
  }
}

function evLabel(id: string): string {
  return id.replace(/^ev[-_]/, '')
}
</script>

<style scoped>
.panel-head {
  font-size: 14px;
  font-weight: 600;
  color: var(--eae-text-primary);
  margin-bottom: var(--eae-space-3);
  display: flex;
  align-items: center;
  gap: var(--eae-space-2);
}

.head-tag {
  margin-left: auto;
}

.claims {
  display: flex;
  flex-direction: column;
  gap: var(--eae-space-2);
  margin-bottom: var(--eae-space-3);
}

.claim {
  display: flex;
  gap: var(--eae-space-2);
  border: 1px solid var(--eae-border);
  border-radius: var(--eae-radius-sm);
  padding: var(--eae-space-2) var(--eae-space-3);
}

.claim-supported {
  border-left: 3px solid var(--eae-success);
}

.claim-unsupported {
  border-left: 3px solid var(--eae-warning);
}

.claim-conflict {
  border-left: 3px solid var(--eae-danger);
}

.claim-mark {
  width: 16px;
  flex: 0 0 auto;
  font-weight: 700;
  color: var(--eae-text-secondary);
}

.claim-supported .claim-mark {
  color: var(--eae-success);
}

.claim-unsupported .claim-mark {
  color: var(--eae-warning);
}

.claim-conflict .claim-mark {
  color: var(--eae-danger);
}

.claim-status-line {
  display: flex;
  gap: var(--eae-space-2);
  font-size: 12px;
  color: var(--eae-text-secondary);
}

.claim-status {
  font-weight: 600;
  color: var(--eae-text-primary);
}

.claim-reason {
  font-size: 12px;
  color: var(--eae-text-secondary);
  margin-top: 2px;
  white-space: pre-wrap;
}

.claim-refs {
  font-size: 11px;
  color: var(--eae-text-muted);
  margin-top: 2px;
}

.subhead {
  font-size: 12px;
  font-weight: 600;
  color: var(--eae-text-secondary);
  margin-bottom: var(--eae-space-2);
}

.evidence {
  margin-bottom: var(--eae-space-3);
}

.ev-row {
  display: flex;
  flex-wrap: wrap;
  gap: var(--eae-space-2);
  align-items: baseline;
  font-size: 12px;
  padding: var(--eae-space-1) 0;
  border-bottom: 1px solid var(--eae-border);
}

.ev-row:last-child {
  border-bottom: none;
}

.ev-type {
  flex: 0 0 auto;
}

.ev-ref {
  color: var(--eae-text-primary);
  word-break: break-all;
}

.ev-score {
  margin-left: auto;
  color: var(--eae-text-muted);
}

.ev-detail {
  width: 100%;
  color: var(--eae-text-secondary);
  font-size: 11px;
  white-space: pre-wrap;
  word-break: break-word;
}

.toolresults {
  margin-bottom: var(--eae-space-3);
}

.toolrow {
  display: flex;
  align-items: center;
  gap: var(--eae-space-2);
  font-size: 12px;
  padding: var(--eae-space-1) 0;
}

.toolok {
  color: var(--eae-success);
}

.toolfail {
  color: var(--eae-danger);
}

.panel-empty-sm {
  color: var(--eae-text-muted);
  font-size: 13px;
  padding: var(--eae-space-2) 0;
}
</style>
