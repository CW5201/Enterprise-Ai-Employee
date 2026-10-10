import client, { apiUrl } from './client'
import type { ReportDetail, ReportRecord } from '@/types'

export async function listReports(): Promise<ReportRecord[]> {
  return client.get<ReportRecord[]>('/api/reports')
}

export async function getReport(id: string): Promise<ReportDetail> {
  return client.get<ReportDetail>(`/api/reports/${id}`)
}

/** Browser-download URL for a stored report (Markdown attachment). */
export function reportDownloadUrl(id: string): string {
  return apiUrl(`/api/reports/${id}/download`)
}
