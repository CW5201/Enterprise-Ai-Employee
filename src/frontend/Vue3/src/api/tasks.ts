import client from './client'
import type { TaskRecord } from '@/types'

export async function listTasks(): Promise<TaskRecord[]> {
  return client.get<TaskRecord[]>('/api/tasks')
}

export async function getTask(id: string): Promise<TaskRecord> {
  return client.get<TaskRecord>(`/api/tasks/${id}`)
}
