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

export interface KnowledgeContent {
  id: string
  title: string
  content: string
}

/** Raw Markdown body of one KB document (GET /api/knowledge/documents/{id}/content). */
export async function getKnowledgeContent(id: string): Promise<KnowledgeContent> {
  return client.get<KnowledgeContent>(`/api/knowledge/documents/${id}/content`)
}
