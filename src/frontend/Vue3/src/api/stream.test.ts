import { afterEach, describe, expect, it, vi } from 'vitest'
import { streamChat } from './stream'
import type { StreamEvent } from '@/types'

function sseResponse(chunks: string[], status = 200): Response {
  const encoder = new TextEncoder()
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk))
      controller.close()
    },
  })
  return new Response(body, { status })
}

function frame(type: string, payload: Record<string, unknown> = {}): string {
  return `data: ${JSON.stringify({ type, timestamp: 't', step: type, status: 'ok', payload })}\n\n`
}

function recorder() {
  const events: StreamEvent[] = []
  const errors: Error[] = []
  let doneCount = 0
  return {
    events,
    errors,
    doneCount: () => doneCount,
    handlers: {
      onEvent: (ev: StreamEvent) => events.push(ev),
      onError: (err: Error) => errors.push(err),
      onDone: () => {
        doneCount += 1
      },
    },
  }
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('streamChat', () => {
  it('hands each SSE frame to onEvent in order', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        sseResponse([frame('task_started'), frame('routing', { route: 'sql' }), frame('completed')]),
      ),
    )
    const rec = recorder()

    await streamChat('查一下销售额', rec.handlers)

    expect(rec.events.map((e) => e.type)).toEqual(['task_started', 'routing', 'completed'])
    expect(rec.events[1].payload.route).toBe('sql')
    expect(rec.errors).toHaveLength(0)
    expect(rec.doneCount()).toBe(1)
  })

  it('reassembles frames split across chunk boundaries', async () => {
    // A frame arrives in three pieces, including a split inside the JSON body.
    const whole = frame('answer', { answer: '合计 12345' })
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        sseResponse([whole.slice(0, 12), whole.slice(12, 30), whole.slice(30)]),
      ),
    )
    const rec = recorder()

    await streamChat('q', rec.handlers)

    expect(rec.events).toHaveLength(1)
    expect(rec.events[0].payload.answer).toBe('合计 12345')
  })

  it('skips malformed frames instead of tearing the stream down', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => sseResponse(['data: {not json}\n\n', frame('answer', { answer: 'ok' })])),
    )
    const rec = recorder()

    await streamChat('q', rec.handlers)

    expect(rec.events.map((e) => e.type)).toEqual(['answer'])
    expect(rec.errors).toHaveLength(0)
  })

  it('reports an HTTP failure through onError and still finishes', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(null, { status: 502 })))
    const rec = recorder()

    await streamChat('q', rec.handlers)

    expect(rec.errors).toHaveLength(1)
    expect(rec.errors[0].message).toContain('502')
    expect(rec.doneCount()).toBe(1)
  })

  it('treats an aborted request as a quiet stop, not an error', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new DOMException('aborted', 'AbortError')
      }),
    )
    const rec = recorder()

    await streamChat('q', rec.handlers)

    expect(rec.errors).toHaveLength(0)
    expect(rec.doneCount()).toBe(1)
  })

  it('reports a network failure that happens before the stream opens', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new TypeError('Failed to fetch')
      }),
    )
    const rec = recorder()

    await streamChat('q', rec.handlers)

    expect(rec.errors).toHaveLength(1)
    expect(rec.errors[0].message).toBe('Failed to fetch')
    expect(rec.doneCount()).toBe(1)
  })

  it('does not report an AbortError raised while reading the body', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        const body = new ReadableStream<Uint8Array>({
          pull() {
            throw new DOMException('aborted', 'AbortError')
          },
        })
        return new Response(body, { status: 200 })
      }),
    )
    const rec = recorder()

    await streamChat('q', rec.handlers)

    expect(rec.errors).toHaveLength(0)
    expect(rec.doneCount()).toBe(1)
  })
})
