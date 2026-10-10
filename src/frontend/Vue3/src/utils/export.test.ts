import { describe, expect, it } from 'vitest'
import { buildResultJson, buildResultMarkdown, resultFilename } from './export'
import type { ChatResponse } from '@/types'

function makeResult(overrides: Partial<ChatResponse> = {}): ChatResponse {
  return {
    request_id: 'req-1',
    answer: '最近三个月销售额合计 12345。',
    intent: 'query',
    route: 'sql',
    sql: 'SELECT sum(amount) FROM fact_sale',
    evidence: [],
    tool_calls: [],
    status: 'completed',
    latency: { total_ms: 1234 },
    sources: [],
    verification: [],
    verification_summary: {},
    guard: {},
    ...overrides,
  }
}

describe('buildResultMarkdown', () => {
  it('renders the task, the answer and the run info', () => {
    const md = buildResultMarkdown(makeResult(), '统计最近三个月销售额')

    expect(md).toContain('# 企业 AI 数字员工任务结果')
    expect(md).toContain('## 任务')
    expect(md).toContain('统计最近三个月销售额')
    expect(md).toContain('最近三个月销售额合计 12345。')
    expect(md).toContain('```sql')
    expect(md).toContain('SELECT sum(amount) FROM fact_sale')
    expect(md).toContain('1.23 s')
    expect(md).toContain('req-1')
  })

  it('omits the sections a run did not produce', () => {
    const md = buildResultMarkdown(makeResult({ sql: '' }))

    expect(md).not.toContain('## 任务')
    expect(md).not.toContain('```sql')
    expect(md).not.toContain('## 证据来源')
    expect(md).not.toContain('## 校验结论')
    expect(md).not.toContain('## 工具调用')
  })

  it('lists evidence with its source and verification verdicts', () => {
    const md = buildResultMarkdown(
      makeResult({
        evidence: [
          {
            evidence_id: 'e1',
            source_type: 'duckdb',
            source_ref: 'fact_sale',
            payload: {},
            metadata: {},
            score: 0.87,
          },
        ],
        verification: [
          {
            claim_id: 'c1',
            supported: true,
            conflict: false,
            status: 'supported',
            support_score: 0.9,
            reason: 'SQL 结果一致',
          },
        ],
        verification_summary: { supported_claims: 1, unsupported_claims: 0, conflicts: 0 },
      }),
    )

    expect(md).toContain('## 证据来源')
    expect(md).toContain('- duckdb（fact_sale），score 0.87')
    expect(md).toContain('## 校验结论')
    expect(md).toContain('- 支持：1')
    expect(md).toContain('- [supported] c1，support 0.9：SQL 结果一致')
  })

  it('marks failed tool calls with their error', () => {
    const md = buildResultMarkdown(
      makeResult({
        tool_calls: [
          {
            tool: 'sql',
            ok: false,
            input: {},
            error: 'DuckDB 只读拒绝',
            duration_ms: 4.2,
            audit_id: 'a1',
          },
        ],
      }),
    )

    expect(md).toContain('## 工具调用')
    expect(md).toContain('- sql — fail（4.2 ms）：DuckDB 只读拒绝')
  })
})

describe('buildResultJson', () => {
  it('round-trips the run together with its question', () => {
    const parsed = JSON.parse(buildResultJson(makeResult(), '统计销售额'))

    expect(parsed.question).toBe('统计销售额')
    expect(parsed.result.answer).toContain('12345')
    expect(parsed.result.request_id).toBe('req-1')
    expect(typeof parsed.exported_at).toBe('string')
  })
})

describe('resultFilename', () => {
  it('stamps the filename with the export time', () => {
    expect(resultFilename('task-result', 'md', new Date('2026-10-09T08:30:00Z'))).toBe(
      'task-result-2026-10-09T08-30-00.md',
    )
  })
})
