import client from './client'
import type { GraphEntity, GraphRelation } from '@/types'

export interface GraphQueryParams {
  search?: string
  entity?: string
  limit?: number
}

export async function listGraphEntities(params?: GraphQueryParams): Promise<GraphEntity[]> {
  const query = new URLSearchParams()
  if (params?.search) query.set('search', params.search)
  if (params?.limit) query.set('limit', String(params.limit))
  const qs = query.toString()
  return client.get<GraphEntity[]>(`/api/graph/entities${qs ? `?${qs}` : ''}`)
}

export async function listGraphRelations(params?: GraphQueryParams): Promise<GraphRelation[]> {
  const query = new URLSearchParams()
  if (params?.entity) query.set('entity', params.entity)
  if (params?.limit) query.set('limit', String(params.limit))
  const qs = query.toString()
  return client.get<GraphRelation[]>(`/api/graph/relations${qs ? `?${qs}` : ''}`)
}
