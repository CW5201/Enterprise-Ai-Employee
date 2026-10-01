import client from './client'
import type { AnalyticsSummary, AnalyticsTrends } from '@/types'

export async function getAnalyticsSummary(): Promise<AnalyticsSummary> {
  return client.get<AnalyticsSummary>('/api/analytics/summary')
}

export async function getAnalyticsTrends(): Promise<AnalyticsTrends> {
  return client.get<AnalyticsTrends>('/api/analytics/trends')
}
