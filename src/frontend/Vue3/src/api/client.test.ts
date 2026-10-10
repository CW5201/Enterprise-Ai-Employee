import { describe, expect, it } from 'vitest'
import { apiUrl } from './client'

describe('apiUrl', () => {
  it('builds a root-relative API path for the fetch-based callers', () => {
    // BASE_URL is '/' in the default (and test) configuration.
    expect(apiUrl('/api/reports/abc/download')).toBe('/api/reports/abc/download')
    expect(apiUrl('api/chat/stream')).toBe('/api/chat/stream')
  })
})
