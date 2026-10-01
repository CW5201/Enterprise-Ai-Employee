import client from './client'
import type { EvaluationSummary } from '@/types'

export async function getEvaluationSummary(): Promise<EvaluationSummary> {
  return client.get<EvaluationSummary>('/api/evaluation/summary')
}
