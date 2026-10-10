import type { ChatResponse } from '@/types'

/**
 * Client-side export helpers.
 *
 * Two export paths exist in the workbench and they are deliberately distinct:
 *
 * - the *server* report (`/api/reports/{id}/download`) is assembled by
 *   ReportTool from the stored run;
 * - these builders export the run that is currently on screen, so the user
 *   can keep the result without depending on the server-side store (which is
 *   in-memory and per-process).
 *
 * Both render only fields the API already returned - nothing is invented
 * here, empty fields are simply omitted.
 */

export function downloadText(
  filename: string,
  text: string,
  mime = 'text/markdown;charset=utf-8',
): void {
  const url = URL.createObjectURL(new Blob([text], { type: mime }))
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(url)
}

/** Timestamped filename, e.g. `task-result-2026-10-09T08-30-00.md`. */
export function resultFilename(prefix: string, ext: string, at: Date = new Date()): string {
  const stamp = at.toISOString().replace(/[:.]/g, '-').slice(0, 19)
  return `${prefix}-${stamp}.${ext}`
}

function fmtMs(ms?: number): string {
  return ms == null ? '—' : `${(ms / 1000).toFixed(2)} s`
}

export function buildResultMarkdown(result: ChatResponse, question = ''): string {
  const lines: string[] = ['# 企业 AI 数字员工任务结果', '']

  if (question) {
    lines.push('## 任务', '', question, '')
  }

  lines.push('## 回答', '', result.answer || '（无回答）', '')

  lines.push(
    '## 执行信息',
    '',
    `- 状态：${result.status || '—'}`,
    `- 意图：${result.intent || '—'}`,
    `- 路由：${result.route || '—'}`,
    `- 耗时：${fmtMs(result.latency?.total_ms)}`,
    `- 请求 ID：${result.request_id || '—'}`,
    '',
  )

  if (result.sql) {
    lines.push('## SQL', '', '```sql', result.sql, '```', '')
  }

  if (result.evidence?.length) {
    lines.push('## 证据来源', '')
    for (const ev of result.evidence) {
      const score = ev.score == null ? '' : `，score ${ev.score}`
      lines.push(`- ${ev.source_type}（${ev.source_ref}）${score}`)
    }
    lines.push('')
  }

  const summary = result.verification_summary ?? {}
  if (result.verification?.length) {
    lines.push(
      '## 校验结论',
      '',
      `- 支持：${summary.supported_claims ?? 0}`,
      `- 不支持：${summary.unsupported_claims ?? 0}`,
      `- 冲突：${summary.conflicts ?? 0}`,
      '',
    )
    for (const v of result.verification) {
      const score = v.support_score == null ? '' : `，support ${v.support_score}`
      lines.push(`- [${v.status}] ${v.claim_id}${score}${v.reason ? `：${v.reason}` : ''}`)
    }
    lines.push('')
  }

  if (result.tool_calls?.length) {
    lines.push('## 工具调用', '')
    for (const tc of result.tool_calls) {
      const err = tc.ok ? '' : `：${tc.error ?? '执行失败'}`
      lines.push(`- ${tc.tool} — ${tc.ok ? 'ok' : 'fail'}（${tc.duration_ms?.toFixed(1)} ms）${err}`)
    }
    lines.push('')
  }

  lines.push('---', '', `*导出时间：${new Date().toLocaleString()}*`, '')
  return lines.join('\n')
}

export function buildResultJson(result: ChatResponse, question = ''): string {
  return JSON.stringify(
    {
      exported_at: new Date().toISOString(),
      question,
      result,
    },
    null,
    2,
  )
}
