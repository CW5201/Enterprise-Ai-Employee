/**
 * Global design tokens and CSS-in-JS reset for the Enterprise SaaS look.
 *
 * Principles (see docs/ROADMAP.md Phase 6 visual spec):
 * - light gray / white surfaces, restrained blue primary
 * - small radii (6px default), subtle borders, minimal shadow
 * - no gradients, no glow, no particles
 * - information density > decoration
 */

const STYLES = `
:root {
  /* surfaces */
  --eae-bg: #f5f7fa;
  --eae-surface: #ffffff;
  --eae-surface-alt: #fafbfc;

  /* text */
  --eae-text-primary: #1f2937;
  --eae-text-secondary: #6b7280;
  --eae-text-muted: #9ca3af;

  /* borders */
  --eae-border: #e5e7eb;
  --eae-border-strong: #d1d5db;

  /* brand / status */
  --eae-primary: #2563eb;
  --eae-primary-light: #eff6ff;
  --eae-success: #16a34a;
  --eae-warning: #f97316;
  --eae-danger: #dc2626;
  --eae-info: #6b7280;

  /* spacing */
  --eae-space-1: 4px;
  --eae-space-2: 8px;
  --eae-space-3: 12px;
  --eae-space-4: 16px;
  --eae-space-5: 20px;
  --eae-space-6: 24px;

  /* radii */
  --eae-radius-sm: 4px;
  --eae-radius: 6px;
  --eae-radius-lg: 8px;

  /* shadow */
  --eae-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);

  /* layout */
  --eae-sidebar-width: 220px;
  --eae-header-height: 52px;
}

* {
  box-sizing: border-box;
}

html,
body,
#app {
  height: 100%;
  margin: 0;
  padding: 0;
}

body {
  background: var(--eae-bg);
  color: var(--eae-text-primary);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue',
    'PingFang SC', 'Microsoft YaHei', sans-serif;
  font-size: 14px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}

/* Element Plus overrides to match the enterprise palette */
.el-card {
  border-radius: var(--eae-radius) !important;
  border: 1px solid var(--eae-border) !important;
  box-shadow: var(--eae-shadow) !important;
}

.el-button {
  border-radius: var(--eae-radius-sm) !important;
}

.el-table {
  --el-table-border-color: var(--eae-border);
}

.el-tag {
  border-radius: var(--eae-radius-sm);
}

/* Page title scale: 20–24px page title, 16–18px section, 14px body */
.page-title {
  font-size: 22px;
  font-weight: 600;
  color: var(--eae-text-primary);
  margin: 0;
  line-height: 1.3;
}

.section-title {
  font-size: 16px;
  font-weight: 600;
  color: var(--eae-text-primary);
  margin: 0;
}

.section-desc {
  font-size: 13px;
  color: var(--eae-text-secondary);
  margin: 2px 0 0;
}
`

export function setupGlobalStyles(): void {
  const styleEl = document.createElement('style')
  styleEl.textContent = STYLES
  document.head.appendChild(styleEl)
}

export default STYLES
