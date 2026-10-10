import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vitest/config'

/**
 * Unit tests for the frontend logic that is worth pinning down without a
 * browser: the SSE frame parser, the result-export builders and the API URL
 * helper.  Component rendering is exercised by the backend API/SSE tests plus
 * the manual smoke run documented in README (Phase 6 前端工作台).
 *
 * The test files are excluded from `tsconfig.app.json`, so the production
 * build (`npm run build` -> vue-tsc) never depends on dev-only test types.
 */
export default defineConfig({
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
})
