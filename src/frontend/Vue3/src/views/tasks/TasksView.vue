<template>
  <div>
    <div class="page-head">
      <div>
        <h1 class="page-title">AI 任务</h1>
        <div class="page-sub">提交企业任务，查看结果、证据与执行过程</div>
      </div>
      <div class="head-actions">
        <el-button v-if="result" size="small" :loading="streaming" @click="() => runStream()">
          重新以流式执行
        </el-button>
      </div>
    </div>

    <div class="task-grid">
      <!-- Left: task conversation -->
      <div class="col col-left">
        <div class="panel">
          <div class="panel-head">任务对话</div>
          <div class="chat-list">
            <div v-for="(m, i) in messages" :key="i" :class="['msg', m.role]">
              <div class="msg-role">{{ m.role === 'user' ? '用户' : 'AI' }}</div>
              <div class="msg-text">{{ m.text }}</div>
            </div>
            <div v-if="!messages.length && !streaming" class="panel-empty">
              输入任务，例如「统计最近三个月销售额并分析增长趋势」
            </div>
          </div>
          <div class="chat-input">
            <el-input
              v-model="question"
              type="textarea"
              :rows="3"
              placeholder="输入企业任务…"
              :disabled="streaming"
            />
            <div class="chat-input-actions">
              <el-select v-model="mode" size="small" class="mode-select" :disabled="streaming">
                <el-option label="流式执行" value="stream" />
                <el-option label="完整请求" value="chat" />
              </el-select>
              <el-button type="primary" :loading="streaming" @click="run">提交</el-button>
            </div>
          </div>
        </div>

        <div class="panel">
          <div class="panel-head">任务历史</div>
          <el-table v-if="tasks.length" :data="tasks" size="small" @row-click="(row: TaskRecord) => loadTask(row)">
            <el-table-column prop="question" label="任务" min-width="160" show-overflow-tooltip />
            <el-table-column label="状态" width="90">
              <template #default="{ row }">
                <el-tag size="small" :type="statusType(row.status)">{{ statusLabel(row.status) }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="route" label="路由" width="90" show-overflow-tooltip />
          </el-table>
          <div v-else-if="historyError" class="panel-empty panel-error">{{ historyError }}</div>
          <div v-else class="panel-empty">暂无任务</div>
        </div>
      </div>

      <!-- Middle: result -->
      <div class="col col-mid">
        <div class="panel">
          <div class="panel-head">
            结果
            <el-tag v-if="result" size="small" class="head-tag" :type="statusType(result.status)">
              {{ statusLabel(result.status) }}
            </el-tag>
            <template v-if="result">
              <el-button size="small" text @click="exportResult('md')">导出 Markdown</el-button>
              <el-button size="small" text @click="exportResult('json')">导出 JSON</el-button>
            </template>
          </div>

          <template v-if="result">
            <div class="kv-grid">
              <div class="kv"><span class="kv-k">意图</span><span class="kv-v">{{ result.intent || '—' }}</span></div>
              <div class="kv"><span class="kv-k">路由</span><span class="kv-v">{{ result.route || '—' }}</span></div>
              <div class="kv">
                <span class="kv-k">耗时</span>
                <span class="kv-v">{{ fmtMs(result.latency?.total_ms) }}</span>
              </div>
            </div>

            <div class="answer-block">
              <div class="subhead">回答</div>
              <div class="answer-text">{{ result.answer || '（无回答）' }}</div>
            </div>

            <div v-if="result.sql" class="answer-block">
              <div class="subhead">SQL</div>
              <pre class="code-block">{{ result.sql }}</pre>
            </div>
          </template>
          <div v-else-if="streaming" class="panel-empty">执行中…</div>
          <div v-else class="panel-empty">提交任务后在此查看结果</div>
        </div>
      </div>

      <!-- Right: execution trace + evidence -->
      <div class="col col-right">
        <div class="panel">
          <div class="panel-head">执行过程</div>
          <ol v-if="trace.length" class="trace">
            <li v-for="(step, i) in trace" :key="i" :class="['trace-step', step.status]">
              <div class="trace-title">{{ step.title }}</div>
              <div v-if="step.detail" class="trace-detail">{{ step.detail }}</div>
            </li>
          </ol>
          <div v-else class="panel-empty">尚无执行记录</div>
        </div>

        <EvidencePanel :result="result" />

        <div v-if="result?.tool_calls?.length" class="panel">
          <div class="panel-head">工具调用</div>
          <div v-for="(tc, i) in result.tool_calls" :key="i" class="tool-row">
            <el-tag size="small" :type="tc.ok ? 'success' : 'danger'">{{ tc.ok ? 'ok' : 'fail' }}</el-tag>
            <span class="tool-name">{{ tc.tool }}</span>
            <span v-if="!tc.ok && tc.error" class="tool-error">{{ tc.error }}</span>
            <span class="tool-ms">{{ tc.duration_ms?.toFixed(1) }} ms</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { postChat } from '@/api/chat'
import { streamChat } from '@/api/stream'
import { listTasks } from '@/api/tasks'
import EvidencePanel from '@/components/EvidencePanel.vue'
import { buildResultJson, buildResultMarkdown, downloadText, resultFilename } from '@/utils/export'
import type { ChatResponse, StreamEvent, TaskRecord } from '@/types'

interface ChatMsg {
  role: 'user' | 'ai'
  text: string
}

interface TraceStep {
  title: string
  detail?: string
  status: 'pending' | 'running' | 'ok' | 'error'
}

const question = ref('')
const mode = ref<'stream' | 'chat'>('stream')
const streaming = ref(false)
const messages = ref<ChatMsg[]>([])
const tasks = ref<TaskRecord[]>([])
const result = ref<ChatResponse | null>(null)
const trace = ref<TraceStep[]>([])
const historyError = ref('')
// The task behind the result currently on screen, used as the subject line
// when the result is exported.
const lastQuestion = ref('')

const abortRef = ref<AbortController | null>(null)

function refreshHistory() {
  listTasks()
    .then((ts) => {
      tasks.value = ts
      historyError.value = ''
    })
    .catch((err) => {
      // Do not swallow: the user must be able to tell "no tasks" from
      // "API unreachable".
      historyError.value = err instanceof Error ? err.message : String(err)
    })
}

function loadTask(row: TaskRecord) {
  // Show the recorded answer in the result pane; full trace re-run via
  // the stream button (history rows store only summary fields).  Build a
  // fresh, fully-populated ChatResponse so no stale fields from a previous
  // result leak into the next run.
  lastQuestion.value = row.question
  result.value = {
    request_id: row.id,
    answer: row.answer ?? '',
    intent: '',
    route: row.route ?? '',
    sql: '',
    evidence: [],
    tool_calls: [],
    status: row.status,
    latency: row.latency_ms != null ? { total_ms: row.latency_ms } : {},
    sources: [],
    verification: [],
    verification_summary: {},
    guard: {},
  } as ChatResponse
  trace.value = [
    { title: '任务提交', detail: row.question, status: 'ok' },
    {
      title: statusType(row.status) === 'danger' ? '执行失败' : '执行完成',
      detail: row.route,
      status: statusType(row.status) === 'danger' ? 'error' : 'ok',
    },
  ]
}

function setTraceFromEvents(evts: { title: string; status: TraceStep['status']; detail?: string }[]) {
  trace.value = evts
}

function buildTrace(result: ChatResponse): TraceStep[] {
  const steps: TraceStep[] = [
    { title: '任务理解', detail: result.intent || undefined, status: 'ok' },
    { title: '动态路由', detail: result.route || undefined, status: 'ok' },
  ]
  for (const tc of result.tool_calls ?? []) {
    steps.push({
      title: `工具：${tc.tool}`,
      detail: tc.ok ? undefined : tc.error ?? '执行失败',
      status: tc.ok ? 'ok' : 'error',
    })
  }
  if (result.evidence?.length) {
    steps.push({ title: '证据收集', detail: `${result.evidence.length} 条证据`, status: 'ok' })
  }
  if (result.verification?.length) {
    const s = result.verification_summary
    steps.push({
      title: '结果验证',
      detail: `支持 ${s?.supported_claims ?? 0} / 不支持 ${s?.unsupported_claims ?? 0} / 冲突 ${s?.conflicts ?? 0}`,
      status: 'ok',
    })
  }
  steps.push({
    title: '完成',
    detail: fmtMs(result.latency?.total_ms),
    // Only a genuinely failed terminal state is an error. The graph reports
    // claims_empty / verified / guarded for successful runs, so comparing
    // against 'completed' alone painted every successful run's last step red.
    status: statusType(result.status) === 'danger' ? 'error' : 'ok',
  })
  return steps
}

function appendStreamStep(ev: StreamEvent) {
  const last = trace.value[trace.value.length - 1]
  const title = streamStepTitle(ev)
  // update-or-append: advance the current step in place
  if (last && last.title === title) {
    last.status = ev.type === 'error' ? 'error' : 'ok'
    last.detail = streamStepDetail(ev, last.detail)
    return
  }
  // mark a still-running previous step as ok
  if (last && last.status === 'running') last.status = 'ok'
  trace.value.push({
    title,
    detail: streamStepDetail(ev, undefined),
    status: ev.type === 'error' ? 'error' : ev.type === 'completed' ? 'ok' : 'running',
  })
}

function streamStepTitle(ev: StreamEvent): string {
  switch (ev.type) {
    case 'task_started':
      return '任务理解'
    case 'routing':
      return '动态路由'
    case 'tool_started':
      return `工具：${(ev.payload.tool as string) ?? ev.step}`
    case 'tool_completed':
      return `工具：${(ev.payload.tool as string) ?? ev.step}`
    case 'verification':
      return '结果验证'
    case 'guard':
      return '安全检查'
    case 'answer':
      return '生成回答'
    case 'completed':
      return '完成'
    case 'error':
      return '执行失败'
    default:
      return ev.step || ev.type
  }
}

function streamStepDetail(ev: StreamEvent, prev: string | undefined): string | undefined {
  const p = ev.payload as Record<string, unknown>
  switch (ev.type) {
    case 'routing':
      return (p.route as string) || undefined
    case 'tool_started':
    case 'tool_completed':
      return (p.tool as string) || prev
    case 'verification': {
      const s = p.verification_summary as { supported_claims?: number; unsupported_claims?: number; conflicts?: number } | undefined
      if (s) return `支持 ${s.supported_claims ?? 0} / 不支持 ${s.unsupported_claims ?? 0} / 冲突 ${s.conflicts ?? 0}`
      return undefined
    }
    case 'answer':
      return (p.status as string) || undefined
    case 'completed':
      return fmtMs((p.latency as Record<string, number>)?.total_ms)
    case 'error':
      return (p.message as string) || '执行出错'
    default:
      return prev
  }
}

async function run() {
  const q = question.value.trim()
  if (!q || streaming.value) return
  lastQuestion.value = q
  messages.value.push({ role: 'user', text: q })
  result.value = null
  trace.value = [{ title: '任务理解', status: 'running' }]
  streaming.value = true

  if (mode.value === 'stream') {
    try {
      await runStream(q)
    } catch (err) {
      // runStream handles its own SSE errors, but anything it throws before
      // the stream opens (or a bug in a handler) would otherwise leave the
      // input box and button disabled forever, because streaming.value = false
      // lives at the end of runStream.
      trace.value.push({
        title: '执行失败',
        detail: err instanceof Error ? err.message : String(err),
        status: 'error',
      })
    } finally {
      streaming.value = false
    }
  } else {
    try {
      const r = await postChat({ message: q })
      result.value = r
      messages.value.push({ role: 'ai', text: r.answer })
      setTraceFromEvents(buildTrace(r))
    } catch (err) {
      trace.value.push({ title: '执行失败', detail: err instanceof Error ? err.message : String(err), status: 'error' })
      messages.value.push({ role: 'ai', text: `执行失败：${err instanceof Error ? err.message : String(err)}` })
    } finally {
      streaming.value = false
      refreshHistory()
    }
  }
}

async function runStream(q?: string) {
  // Vue passes the MouseEvent as the first argument for a bare
  // @click="runStream" binding, and `??` only falls back on null/undefined.
  // Without the typeof check the payload becomes JSON.stringify({}) -> 422.
  const questionText = typeof q === 'string' ? q : question.value.trim()
  if (!questionText) return
  lastQuestion.value = questionText
  // Reset here as well as in run(): the "重新以流式执行" button calls this
  // function directly, and without a reset the previous trace keeps appending
  // and the result panel keeps showing the old run until the new one lands.
  result.value = null
  trace.value = [{ title: '任务理解', status: 'running' }]
  streaming.value = true
  abortRef.value?.abort()
  const ac = new AbortController()
  abortRef.value = ac

  let finalState: ChatResponse | null = null

  await streamChat(
    questionText,
    {
      onEvent: (ev: StreamEvent) => {
        if (ev.type === 'answer') {
          const p = ev.payload as {
            answer?: string
            status?: string
            route?: string
            intent?: string
            sql?: string
            latency?: Record<string, number>
          }
          // The answer node's delta carries intent / sql / evidence /
          // tool_calls now (backend _safe_payload), so keep what arrived
          // instead of blanking it.
          finalState = {
            ...(finalState ?? ({} as ChatResponse)),
            answer: p.answer ?? finalState?.answer ?? '',
            status: p.status ?? '',
            route: p.route ?? finalState?.route ?? '',
            intent: p.intent ?? finalState?.intent ?? '',
            sql: p.sql ?? finalState?.sql ?? '',
            latency: p.latency ?? finalState?.latency ?? {},
          }
          return
        }
        if (ev.type === 'completed') {
          const p = ev.payload as {
            answer?: string
            status?: string
            route?: string
            intent?: string
            sql?: string
            latency?: Record<string, number>
            evidence?: ChatResponse['evidence']
            tool_calls?: ChatResponse['tool_calls']
            verification?: ChatResponse['verification']
            sources?: ChatResponse['sources']
            guard?: ChatResponse['guard']
            verification_summary?: ChatResponse['verification_summary']
          }
          // The completed event now carries the full folded final state, so
          // the streaming panel can show the same depth as the one-shot
          // /api/chat response (evidence / tool_calls / verification).
          finalState = {
            ...(finalState ?? ({} as ChatResponse)),
            answer: p.answer ?? finalState?.answer ?? '',
            status: p.status ?? 'completed',
            route: p.route ?? finalState?.route ?? '',
            intent: p.intent ?? finalState?.intent ?? '',
            sql: p.sql ?? finalState?.sql ?? '',
            latency: p.latency ?? finalState?.latency ?? {},
            evidence: p.evidence ?? finalState?.evidence ?? [],
            tool_calls: p.tool_calls ?? finalState?.tool_calls ?? [],
            verification: p.verification ?? finalState?.verification ?? [],
            sources: p.sources ?? finalState?.sources ?? [],
            guard: p.guard ?? finalState?.guard ?? {},
            verification_summary: p.verification_summary ?? finalState?.verification_summary ?? {},
          }
          return
        }
        if (ev.type === 'error') {
          // The backend sends exactly one `error` event and no `completed`,
          // so without this branch the result / evidence / tool panels stay
          // empty and the trace is left dangling.  Materialise a minimal
          // failed response so the UI reflects the failure.
          const p = ev.payload as { message?: string }
          trace.value.push({ title: '执行失败', detail: p.message ?? '管道异常', status: 'error' })
          finalState = {
            ...(finalState ?? ({} as ChatResponse)),
            status: 'failed',
            answer: finalState?.answer ?? '',
            route: finalState?.route ?? '',
            intent: finalState?.intent ?? '',
            sql: finalState?.sql ?? '',
            latency: finalState?.latency ?? {},
            evidence: finalState?.evidence ?? [],
            tool_calls: finalState?.tool_calls ?? [],
            verification: finalState?.verification ?? [],
            sources: finalState?.sources ?? [],
            guard: finalState?.guard ?? {},
            verification_summary: finalState?.verification_summary ?? {},
          }
          appendStreamStep(ev)
          return
        }
        appendStreamStep(ev)
      },
      onError: (err: Error) => {
        // Network-level failure (fetch rejected / stream torn down) — also
        // materialise a failed result so the panels don't stay blank.
        trace.value.push({ title: '执行失败', detail: err.message, status: 'error' })
        finalState = {
          ...(finalState ?? ({} as ChatResponse)),
          status: 'failed',
          answer: finalState?.answer ?? '',
          route: finalState?.route ?? '',
          latency: finalState?.latency ?? {},
          evidence: [],
          tool_calls: [],
          verification: [],
          sources: [],
          guard: {},
          verification_summary: {},
        }
      },
      onDone: () => {
        if (finalState) {
          result.value = finalState
          const failText =
            finalState.status === 'failed'
              ? '执行失败，请查看执行过程面板。'
              : finalState.answer || '（无回答）'
          messages.value.push({ role: 'ai', text: failText })
          // finalState is only populated by the 'answer' event, which does not
          // carry verification_summary; if the stream closes before the
          // 'completed' frame this is undefined and `s.supported_claims`
          // would throw, leaving streaming.value stuck at true.
          const s = finalState.verification_summary ?? {}
          const vStep = trace.value.find((t) => t.title === '结果验证')
          if (vStep) {
            vStep.status = 'ok'
            vStep.detail = `支持 ${s.supported_claims ?? 0} / 不支持 ${s.unsupported_claims ?? 0} / 冲突 ${s.conflicts ?? 0}`
          }
        }
        streaming.value = false
        refreshHistory()
      },
    },
    ac.signal,
  )
}

function fmtMs(ms?: number): string {
  return ms == null ? '—' : `${(ms / 1000).toFixed(2)} s`
}

/**
 * Export the result on screen.  The server also stores a report per run
 * (报告中心 → 下载); this path exports the run the user is looking at, so a
 * result stays exportable even if the API process was restarted.
 */
function exportResult(kind: 'md' | 'json') {
  const current = result.value
  if (!current) return
  if (kind === 'md') {
    downloadText(
      resultFilename('task-result', 'md'),
      buildResultMarkdown(current, lastQuestion.value),
    )
  } else {
    downloadText(
      resultFilename('task-result', 'json'),
      buildResultJson(current, lastQuestion.value),
      'application/json;charset=utf-8',
    )
  }
}

function statusType(status: string): 'success' | 'danger' | 'warning' | 'info' {
  const s = (status ?? '').toLowerCase()
  // Terminal states the graph actually produces (see answer_guard /
  // claim_extraction / answer_generation). A run that answered is a success
  // even when it carried no claims — otherwise every successful task shows
  // up as a gray "info" chip because its status is claims_empty / verified.
  if (
    [
      'completed',
      'ok',
      'success',
      'verified',
      'guarded',
      'executed',
      'claims_extracted',
      'claims_empty',
      'completed_no_evidence',
      'completed_fallback',
    ].includes(s)
  )
    return 'success'
  if (
    [
      'failed',
      'error',
      'guard_blocked',
      'execution_all_failed',
      'execution_no_plan',
      'claims_extraction_failed',
    ].includes(s)
  )
    return 'danger'
  if (s === 'running' || s === 'pending' || s.startsWith('clarif')) return 'warning'
  return 'info'
}

/** Human label for the pipeline's internal terminal status codes. */
function statusLabel(status: string): string {
  const map: Record<string, string> = {
    completed: '完成',
    ok: '完成',
    success: '完成',
    verified: '完成（已验证）',
    guarded: '完成（含待核实）',
    executed: '完成',
    claims_extracted: '完成',
    claims_empty: '完成（无 claim）',
    completed_no_evidence: '完成（无证据）',
    completed_fallback: '完成（兜底回答）',
    guard_blocked: '已阻止',
    execution_all_failed: '执行失败',
    execution_no_plan: '无法执行',
    claims_extraction_failed: '验证失败',
    failed: '失败',
    error: '错误',
    running: '执行中',
    pending: '等待中',
    clarification: '待澄清',
  }
  const s = (status ?? '').toLowerCase()
  if (map[s]) return map[s]
  if (s.startsWith('clarif')) return '待澄清'
  return status || '—'
}

onMounted(refreshHistory)
</script>

<style scoped>
.task-grid {
  display: grid;
  /* The answer pane is the primary content, so it — not the chat column —
     gets the fluid track. Side panes keep min/max bounds so the middle never
     drops below a readable width. */
  grid-template-columns: minmax(280px, 340px) minmax(420px, 1fr) minmax(240px, 300px);
  gap: var(--eae-space-4);
  align-items: start;
}

/* Content width = viewport - 220px sidebar - 40px padding, so the 3-pane
   layout is only kept when the panes still fit (viewport >= 1300px). */
@media (max-width: 1299px) {
  .task-grid {
    grid-template-columns: minmax(300px, 360px) minmax(420px, 1fr);
  }

  .col-right {
    grid-column: 1 / -1;
  }
}

@media (max-width: 999px) {
  .task-grid {
    grid-template-columns: 1fr;
  }

  .col-right {
    grid-column: auto;
  }
}

.col {
  min-width: 0;
}

.head-tag {
  margin-left: auto;
}

/* chat */
.chat-list {
  display: flex;
  flex-direction: column;
  gap: var(--eae-space-3);
  max-height: 280px;
  overflow-y: auto;
  margin-bottom: var(--eae-space-3);
}

.msg {
  border: 1px solid var(--eae-border);
  border-radius: var(--eae-radius-sm);
  padding: var(--eae-space-2) var(--eae-space-3);
}

.msg-user {
  background: var(--eae-surface-alt);
}

.msg-role {
  font-size: 11px;
  color: var(--eae-text-muted);
  margin-bottom: 2px;
}

.msg-text {
  font-size: 13px;
  white-space: pre-wrap;
  word-break: break-word;
}

.chat-input-actions {
  display: flex;
  gap: var(--eae-space-2);
  margin-top: var(--eae-space-2);
}

.mode-select {
  width: 110px;
}

/* kv + answer */
.kv-grid {
  display: flex;
  flex-wrap: wrap;
  gap: var(--eae-space-3);
  margin-bottom: var(--eae-space-3);
}

.kv {
  display: flex;
  gap: var(--eae-space-2);
  font-size: 13px;
}

.kv-k {
  color: var(--eae-text-muted);
}

.kv-v {
  color: var(--eae-text-primary);
  font-weight: 500;
}

.answer-block {
  margin-bottom: var(--eae-space-3);
}

.subhead {
  font-size: 12px;
  font-weight: 600;
  color: var(--eae-text-secondary);
  margin-bottom: 6px;
}

.answer-text {
  font-size: 13px;
  line-height: 1.7;
  white-space: pre-wrap;
}

.code-block {
  background: var(--eae-surface-alt);
  border: 1px solid var(--eae-border);
  border-radius: var(--eae-radius-sm);
  padding: var(--eae-space-3);
  font-size: 12px;
  overflow-x: auto;
  white-space: pre;
}

/* trace */
.trace {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
}

.trace-step {
  position: relative;
  padding: 0 0 var(--eae-space-4) 18px;
}

.trace-step:not(:last-child)::before {
  content: '';
  position: absolute;
  left: 5px;
  top: 14px;
  bottom: 0;
  width: 1px;
  background: var(--eae-border);
}

.trace-step::after {
  content: '';
  position: absolute;
  left: 0;
  top: 4px;
  width: 11px;
  height: 11px;
  border-radius: 50%;
  background: var(--eae-border-strong);
  border: 2px solid var(--eae-surface);
  box-shadow: 0 0 0 1px var(--eae-border);
}

.trace-step.running::after {
  background: var(--eae-primary);
}

.trace-step.ok::after {
  background: var(--eae-success);
}

.trace-step.error::after {
  background: var(--eae-danger);
}

.trace-title {
  font-size: 13px;
  font-weight: 500;
  color: var(--eae-text-primary);
}

.trace-detail {
  font-size: 12px;
  color: var(--eae-text-secondary);
  margin-top: 2px;
  word-break: break-word;
}

/* tool rows */
.tool-row {
  display: flex;
  align-items: center;
  gap: var(--eae-space-2);
  padding: var(--eae-space-2) 0;
  border-bottom: 1px solid var(--eae-border);
  font-size: 13px;
}

.tool-row:last-child {
  border-bottom: none;
}

.tool-name {
  font-weight: 500;
}

.tool-error {
  color: var(--eae-danger);
  font-size: 12px;
  word-break: break-all;
}

.tool-ms {
  margin-left: auto;
  color: var(--eae-text-muted);
  font-size: 12px;
}
</style>
