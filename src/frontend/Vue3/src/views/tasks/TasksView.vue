<template>
  <div class="page">
    <div class="page-head">
      <div>
        <h1 class="page-title">AI 任务</h1>
        <div class="page-sub">提交企业任务，查看结果、证据与执行过程</div>
      </div>
      <div class="head-actions">
        <el-button v-if="result" size="small" :loading="streaming" @click="runStream">
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
                <el-tag size="small" :type="statusType(row.status)">{{ row.status }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="route" label="路由" width="90" show-overflow-tooltip />
          </el-table>
          <div v-else class="panel-empty">暂无任务</div>
        </div>
      </div>

      <!-- Middle: result -->
      <div class="col col-mid">
        <div class="panel">
          <div class="panel-head">
            结果
            <el-tag v-if="result" size="small" class="head-tag" :type="statusType(result.status)">
              {{ result.status }}
            </el-tag>
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

      <!-- Right: execution trace -->
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

        <div v-if="result?.tool_calls?.length" class="panel">
          <div class="panel-head">工具调用</div>
          <div v-for="(tc, i) in result.tool_calls" :key="i" class="tool-row">
            <el-tag size="small" :type="tc.ok ? 'success' : 'danger'">{{ tc.ok ? 'ok' : 'fail' }}</el-tag>
            <span class="tool-name">{{ tc.tool }}</span>
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

const abortRef = ref<AbortController | null>(null)

function refreshHistory() {
  listTasks().then((ts) => (tasks.value = ts)).catch(() => undefined)
}

function loadTask(row: TaskRecord) {
  // Show the recorded answer in the result pane; full trace re-run via
  // the stream button (history rows store only summary fields).
  result.value = {
    ...(result.value ?? ({} as ChatResponse)),
    answer: row.answer,
    route: row.route,
    status: row.status,
    latency: { total_ms: row.latency_ms ?? 0 },
  } as ChatResponse
  trace.value = [
    { title: '任务提交', detail: row.question, status: 'ok' },
    { title: row.status === 'completed' ? '执行完成' : '执行失败', detail: row.route, status: row.status === 'completed' ? 'ok' : 'error' },
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
  steps.push({ title: '完成', detail: fmtMs(result.latency?.total_ms), status: result.status === 'completed' ? 'ok' : 'error' })
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
  messages.value.push({ role: 'user', text: q })
  result.value = null
  trace.value = [{ title: '任务理解', status: 'running' }]
  streaming.value = true

  if (mode.value === 'stream') {
    await runStream(q)
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
  const questionText = q ?? question.value.trim()
  if (!questionText) return
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
          const p = ev.payload as { answer?: string; status?: string; route?: string; latency?: Record<string, number> }
          finalState = {
            ...(finalState ?? ({} as ChatResponse)),
            answer: p.answer ?? '',
            status: p.status ?? '',
            route: p.route ?? '',
            intent: '',
            sql: '',
            evidence: [],
            tool_calls: [],
            latency: p.latency ?? {},
            sources: [],
            verification: [],
            verification_summary: {},
            guard: {},
          }
          return
        }
        if (ev.type === 'completed') {
          const p = ev.payload as {
            answer?: string
            status?: string
            route?: string
            latency?: Record<string, number>
            verification_summary?: ChatResponse['verification_summary']
          }
          finalState = {
            ...(finalState ?? ({} as ChatResponse)),
            answer: p.answer ?? finalState?.answer ?? '',
            status: p.status ?? 'completed',
            route: p.route ?? '',
            latency: p.latency ?? {},
            verification_summary: p.verification_summary ?? {},
          }
          return
        }
        appendStreamStep(ev)
      },
      onError: (err: Error) => {
        trace.value.push({ title: '执行失败', detail: err.message, status: 'error' })
      },
      onDone: () => {
        if (finalState) {
          result.value = finalState
          messages.value.push({ role: 'ai', text: finalState.answer || '（无回答）' })
          const s = finalState.verification_summary
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

function statusType(status: string): 'success' | 'danger' | 'warning' | 'info' {
  const s = (status ?? '').toLowerCase()
  if (['completed', 'ok', 'success'].includes(s)) return 'success'
  if (['failed', 'error'].includes(s)) return 'danger'
  if (['running', 'pending'].includes(s)) return 'warning'
  return 'info'
}

onMounted(refreshHistory)
</script>

<style scoped>
.page-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  margin-bottom: var(--eae-space-4);
}

.head-actions {
  margin-top: 4px;
}

.page-sub {
  font-size: 13px;
  color: var(--eae-text-secondary);
}

.task-grid {
  display: grid;
  grid-template-columns: 340px 1fr 280px;
  gap: var(--eae-space-4);
  align-items: start;
}

@media (max-width: 1200px) {
  .task-grid {
    grid-template-columns: 1fr;
  }
}

.col {
  min-width: 0;
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
  color: var(--eae-text-primary);
  margin-bottom: var(--eae-space-3);
  display: flex;
  align-items: center;
  gap: var(--eae-space-2);
}

.head-tag {
  margin-left: auto;
}

.panel-empty {
  padding: var(--eae-space-4);
  color: var(--eae-text-muted);
  font-size: 13px;
  text-align: center;
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

.tool-ms {
  margin-left: auto;
  color: var(--eae-text-muted);
  font-size: 12px;
}
</style>
