import client from './client'
import type { HealthResponse } from '@/types'

export async function getHealth(): Promise<HealthResponse> {
  return client.get<HealthResponse>('/health')
}
