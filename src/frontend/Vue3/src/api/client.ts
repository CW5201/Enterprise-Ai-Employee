import axios from 'axios'
import type { AxiosError, AxiosInstance } from 'axios'

const DEFAULT_TIMEOUT_MS = 60_000

class ApiClient {
  private readonly http: AxiosInstance

  constructor(baseURL: string, timeout: number = DEFAULT_TIMEOUT_MS) {
    this.http = axios.create({ baseURL, timeout })
  }

  async get<T>(url: string): Promise<T> {
    try {
      const res = await this.http.get<T>(url)
      return res.data
    } catch (err) {
      throw normalizeError(err)
    }
  }

  async post<T>(url: string, data?: unknown): Promise<T> {
    try {
      const res = await this.http.post<T>(url, data)
      return res.data
    } catch (err) {
      throw normalizeError(err)
    }
  }
}

export class ApiError extends Error {
  readonly status?: number
  readonly detail?: string

  constructor(message: string, status?: number, detail?: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

function normalizeError(err: unknown): Error {
  if (axios.isAxiosError(err)) {
    const axiosErr = err as AxiosError<{ detail?: string }>
    const status = axiosErr.response?.status
    const detail = axiosErr.response?.data?.detail
    return new ApiError(
      status ? `请求失败（${status}）：${detail ?? axiosErr.message}` : `网络错误：${axiosErr.message}`,
      status,
      detail,
    )
  }
  return err instanceof Error ? err : new Error(String(err))
}

/**
 * Absolute-from-root API path that honours the app's base URL.
 *
 * Used by the places that bypass axios - `fetch` for SSE and for plain
 * browser downloads (`<a href>`) - so they resolve to the same origin the
 * client talks to instead of hardcoding "/api".
 */
export function apiUrl(path: string): string {
  const base = import.meta.env.BASE_URL || '/'
  return `${base.replace(/\/$/, '')}/${path.replace(/^\//, '')}`
}

const client = new ApiClient(import.meta.env.BASE_URL || '')

export default client
