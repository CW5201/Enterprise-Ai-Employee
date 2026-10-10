import { apiUrl } from './client'
import type { StreamEvent } from '@/types'

/**
 * SSE client for POST /api/chat/stream.
 *
 * Browsers' EventSource only supports GET, so the fetch + ReadableStream
 * approach is used: POST the question, parse `data: {json}` frames, and
 * hand structured events to the onEvent callback in real time.
 * No polling, no mock.
 */
export interface StreamHandlers {
  onEvent: (ev: StreamEvent) => void
  onError: (err: Error) => void
  onDone: () => void
}

export async function streamChat(
  question: string,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  let resp: Response
  try {
    resp = await fetch(apiUrl('/api/chat/stream'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: question }),
      signal,
    })
  } catch (err) {
    // A user-cancelled run is not a failure: the result pane already shows
    // the partial state, so only real network errors are reported.
    if ((err as Error).name !== 'AbortError') {
      handlers.onError(err instanceof Error ? err : new Error(String(err)))
    }
    handlers.onDone()
    return
  }

  if (!resp.ok || !resp.body) {
    handlers.onError(new Error(`stream failed (${resp.status})`))
    handlers.onDone()
    return
  }

  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      // SSE frames are separated by a blank line.
      let sep: number
      while ((sep = buffer.indexOf('\n\n')) !== -1) {
        const frame = buffer.slice(0, sep)
        buffer = buffer.slice(sep + 2)
        for (const line of frame.split('\n')) {
          if (!line.startsWith('data:')) continue
          const raw = line.slice(5).trim()
          if (!raw) continue
          try {
            handlers.onEvent(JSON.parse(raw) as StreamEvent)
          } catch {
            // Malformed frame: skip, do not crash the whole stream.
          }
        }
      }
    }
  } catch (err) {
    if ((err as Error).name !== 'AbortError') {
      handlers.onError(err instanceof Error ? err : new Error(String(err)))
    }
  } finally {
    handlers.onDone()
  }
}
