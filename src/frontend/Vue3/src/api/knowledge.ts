import client from './client'
import type { KnowledgeDocument } from '@/types'

export interface KnowledgeListParams {
  q?: string
  department?: string
}

export async function listKnowledgeDocuments(params?: KnowledgeListParams): Promise<KnowledgeDocument[]> {
  const query = new URLSearchParams()
  if (params?.q) query.set('q', params.q)
  if (params?.department) query.set('department', params.department)
  const qs = query.toString()
  return client.get<KnowledgeDocument[]>(`/api/knowledge/documents${qs ? `?${qs}` : ''}`)
}

export async function getKnowledgeDocument(id: string): Promise<KnowledgeDocument> {
  return client.get<KnowledgeDocument>(`/api/knowledge/documents/${id}`)
}
