import client from './client'
import type { ChatResponse } from '@/types'

export interface ChatRequest {
  message: string
  request_id?: string | null
}

export async function postChat(body: ChatRequest): Promise<ChatResponse> {
  return client.post<ChatResponse>('/api/chat', body)
}
