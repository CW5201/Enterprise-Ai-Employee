/** Shared response types, mirroring src/core/state.py + Phase 4/5 surfaces. */

// --- Tool / evidence ---

export interface ToolCallRecord {
  tool: string
  ok: boolean
  input: Record<string, unknown>
  error?: string | null
  evidence_ref?: string | null
  duration_ms: number
  audit_id: string
}

export interface SQLResultPayload {
  sql: string
  columns: string[]
  rows: unknown[][]
  row_count: number
  execution_time_ms: number
  truncated: boolean
}

export interface EvidenceItem {
  evidence_id: string
  source_type: 'duckdb' | 'milvus' | 'fake' | 'rag' | 'sql' | 'kg' | 'analysis' | string
  source_ref: string
  content?: string
  payload: Record<string, unknown>
  score?: number | null
  metadata: Record<string, unknown>
}

// --- Chat / pipeline ---

export type VerificationStatus = 'supported' | 'unsupported' | 'conflict'

export interface VerificationResult {
  claim_id: string
  claim_type?: string
  supported: boolean
  support_score?: number
  evidence_ids?: string[]
  conflict: boolean
  status: VerificationStatus
  verifier_type?: string
  reason?: string
  computed_value?: unknown
  claimed_value?: unknown
}

export interface VerificationSummary {
  supported_claims?: number
  unsupported_claims?: number
  conflicts?: number
  total?: number
  [key: string]: number | undefined
}

export interface ChatResponse {
  request_id: string
  answer: string
  intent: string
  route: string
  sql: string
  evidence: EvidenceItem[]
  tool_calls: ToolCallRecord[]
  status: string
  latency: Record<string, number>
  sources: Array<Record<string, unknown>>
  verification: VerificationResult[]
  verification_summary: VerificationSummary
  guard: Record<string, unknown>
}

// --- SSE stream events ---

export type StreamEventType =
  | 'task_started'
  | 'routing'
  | 'tool_started'
  | 'tool_completed'
  | 'verification'
  | 'answer'
  | 'guard'
  | 'completed'
  | 'error'

export interface StreamEvent {
  type: StreamEventType | string
  timestamp: string
  step: string
  status: string
  payload: Record<string, unknown>
}

// --- Health ---

export interface HealthResponse {
  status: string
  llm?: string
  embedding?: string
  milvus?: string
  milvus_details?: Record<string, unknown>
  database?: string
}

// --- Tasks ---

export type TaskStatus =
  | 'pending'
  | 'running'
  | 'completed'
  | 'failed'
  | string

export interface TaskRecord {
  id: string
  question: string
  status: TaskStatus
  created_at: string
  completed_at?: string | null
  latency_ms?: number | null
  route?: string
  answer?: string
}

export interface TaskCreateBody {
  question: string
}

// --- Knowledge ---

export interface KnowledgeDocument {
  id: string
  title: string
  department: string
  source: string
  effective_date?: string | null
  synthetic: boolean
}

// --- Reports ---
// Reports are assembled server-side from a completed run (answer + evidence +
// verification verdicts) and stored by /api/reports.

export interface ReportRecord {
  id: string
  title: string
  question: string
  request_id: string
  created_at: string
  format: string
  content_chars: number
  evidence_count: number
  supported_claims: number
  unsupported_claims: number
  conflicts: number
}

export interface ReportDetail extends ReportRecord {
  markdown: string
}

// --- Graph ---

export interface GraphEntity {
  id: string
  name: string
  type: string
  properties?: Record<string, unknown>
}

export interface GraphRelation {
  source: string
  type: string
  target: string
}

// --- Evaluation ---

export interface EvaluationMetrics {
  tasks?: number
  task_success_rate?: number
  answer_exact?: number
  route_accuracy?: number
  tool_f1?: number
  plan_exact_match?: number
  evidence_attribution_accuracy?: number
  error_rate?: number
  llm_calls_avg?: number
  latency_total_ms_mean?: number
  latency_total_ms_p50?: number
  latency_total_ms_p95?: number
  [key: string]: number | undefined
}

export interface EvaluationEntry {
  metrics: EvaluationMetrics
  config: Record<string, unknown>
}

export interface EvaluationSummary {
  baselines: Record<string, EvaluationEntry>
  ablations: Record<string, EvaluationEntry>
  verification: Record<string, Record<string, number>>
  available: boolean
  report_path?: string | null
  report_excerpt?: string | null
}

// --- Analytics ---

export interface AnalyticsKPI {
  label: string
  value: number | string
  unit?: string
  delta?: number | null
}

export interface AnalyticsSummary {
  available: boolean
  reason?: string
  kpis?: AnalyticsKPI[]
}

export interface TrendPoint {
  month: string
  total: number
}

export interface TopEntry {
  name: string
  total: number
}

export interface AnalyticsTrends {
  available: boolean
  reason?: string
  monthly_sales?: TrendPoint[]
  top_suppliers?: TopEntry[]
  top_customers?: TopEntry[]
  order_stats?: {
    total: number
    credit_notes: number
    first_invoice: string
    last_invoice: string
  }
}
