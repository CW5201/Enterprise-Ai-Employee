import { createRouter, createWebHistory } from 'vue-router'

import AppLayout from '@/layouts/AppLayout.vue'
import DashboardView from '@/views/dashboard/DashboardView.vue'
import TasksView from '@/views/tasks/TasksView.vue'
import AnalyticsView from '@/views/analytics/AnalyticsView.vue'
import KnowledgeView from '@/views/knowledge/KnowledgeView.vue'
import GraphView from '@/views/graph/GraphView.vue'
import ReportsView from '@/views/reports/ReportsView.vue'
import EvaluationView from '@/views/evaluation/EvaluationView.vue'
import SettingsView from '@/views/settings/SettingsView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      component: AppLayout,
      redirect: '/dashboard',
      children: [
        {
          path: 'dashboard',
          name: 'dashboard',
          component: DashboardView,
          meta: { title: '工作台' },
        },
        {
          path: 'tasks',
          name: 'tasks',
          component: TasksView,
          meta: { title: 'AI 任务' },
        },
        {
          path: 'analytics',
          name: 'analytics',
          component: AnalyticsView,
          meta: { title: '数据分析' },
        },
        {
          path: 'knowledge',
          name: 'knowledge',
          component: KnowledgeView,
          meta: { title: '企业知识' },
        },
        {
          path: 'graph',
          name: 'graph',
          component: GraphView,
          meta: { title: '知识图谱' },
        },
        {
          path: 'reports',
          name: 'reports',
          component: ReportsView,
          meta: { title: '报告中心' },
        },
        {
          path: 'evaluation',
          name: 'evaluation',
          component: EvaluationView,
          meta: { title: '实验评测' },
        },
        {
          path: 'settings',
          name: 'settings',
          component: SettingsView,
          meta: { title: '系统设置' },
        },
      ],
    },
  ],
})

router.afterEach((to) => {
  const title = to.meta?.title as string | undefined
  document.title = title ? `${title} · Enterprise AI Employee` : 'Enterprise AI Employee'
})

export default router
