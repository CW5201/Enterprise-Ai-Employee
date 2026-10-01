import client from './client'
import type { TaskRecord } from '@/types'

export interface TaskCreateBody {
  question: string
}

export async function listTasks(): Promise<TaskRecord[]> {
  return client.get<TaskRecord[]>('/api/tasks')
}

export async function getTask(id: string): Promise<TaskRecord> {
  return client.get<TaskRecord>(`/api/tasks/${id}`)
}

export async function createTask(body: TaskCreateBody): Promise<TaskRecord> {
  return client.post<TaskRecord>('/api/tasks', body)
}
