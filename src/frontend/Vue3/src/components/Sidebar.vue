<template>
  <el-aside class="app-aside" :class="{ 'is-collapsed': layout.collapsed }">
    <div class="sidebar-brand">
      <span class="sidebar-logo">EAE</span>
      <div class="sidebar-brand-text">
        <div class="sidebar-brand-title">Enterprise AI Employee</div>
        <div class="sidebar-brand-sub">企业工作台</div>
      </div>
    </div>
    <el-menu
      :default-active="activeRoute"
      router
      class="sidebar-menu"
      :collapse="layout.collapsed"
    >
      <el-menu-item index="/dashboard">
        <el-icon><Odometer /></el-icon>
        <span>工作台</span>
      </el-menu-item>
      <el-menu-item index="/tasks">
        <el-icon><List /></el-icon>
        <span>AI 任务</span>
      </el-menu-item>
      <el-menu-item index="/analytics">
        <el-icon><DataAnalysis /></el-icon>
        <span>数据分析</span>
      </el-menu-item>
      <el-menu-item index="/knowledge">
        <el-icon><Collection /></el-icon>
        <span>企业知识</span>
      </el-menu-item>
      <el-menu-item index="/graph">
        <el-icon><Connection /></el-icon>
        <span>知识图谱</span>
      </el-menu-item>
      <el-menu-item index="/reports">
        <el-icon><Document /></el-icon>
        <span>报告中心</span>
      </el-menu-item>
      <el-menu-item index="/evaluation">
        <el-icon><Trophy /></el-icon>
        <span>实验评测</span>
      </el-menu-item>
      <el-menu-item index="/settings">
        <el-icon><Setting /></el-icon>
        <span>系统设置</span>
      </el-menu-item>
    </el-menu>
  </el-aside>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import {
  Collection,
  Connection,
  DataAnalysis,
  Document,
  List,
  Odometer,
  Setting,
  Trophy,
} from '@element-plus/icons-vue'
import { useLayoutStore } from '@/stores/layout'

const route = useRoute()
const layout = useLayoutStore()
const activeRoute = computed(() => route.path)
</script>

<style scoped>
.app-aside {
  width: var(--eae-sidebar-width);
  flex-shrink: 0;
  overflow-x: hidden;
  overflow-y: auto;
  background: var(--eae-surface);
  border-right: 1px solid var(--eae-border);
  display: flex;
  flex-direction: column;
  transition: width 0.2s ease;
}

.app-aside.is-collapsed {
  width: var(--eae-sidebar-collapsed-width);
}

.sidebar-brand {
  display: flex;
  align-items: center;
  gap: var(--eae-space-2);
  padding: var(--eae-space-4);
  border-bottom: 1px solid var(--eae-border);
}

.sidebar-logo {
  display: none;
  width: 28px;
  height: 28px;
  border-radius: var(--eae-radius-sm);
  background: var(--eae-primary);
  color: #fff;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.5px;
  align-items: center;
  justify-content: center;
  flex: 0 0 auto;
}

.sidebar-brand-text {
  min-width: 0;
}

.sidebar-brand-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--eae-text-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.sidebar-brand-sub {
  font-size: 12px;
  color: var(--eae-text-secondary);
  margin-top: 2px;
}

.app-aside.is-collapsed .sidebar-brand {
  justify-content: center;
  padding-left: 0;
  padding-right: 0;
}

.app-aside.is-collapsed .sidebar-logo {
  display: flex;
}

.app-aside.is-collapsed .sidebar-brand-text {
  display: none;
}

.sidebar-menu {
  border-right: none;
  flex: 1;
}

.sidebar-menu:not(.el-menu--collapse) {
  width: 100%;
}

.sidebar-menu .el-menu-item.is-active {
  background: var(--eae-primary-light);
  color: var(--eae-primary);
}
</style>
