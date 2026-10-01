<template>
  <el-aside :width="SIDEBAR_WIDTH" class="app-aside">
    <div class="sidebar-brand">
      <div class="sidebar-brand-title">Enterprise AI Employee</div>
      <div class="sidebar-brand-sub">企业工作台</div>
    </div>
    <el-menu :default-active="activeRoute" router class="sidebar-menu" :collapse="collapsed">
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

const SIDEBAR_WIDTH = '220px'
const route = useRoute()
const collapsed = computed(() => false)
const activeRoute = computed(() => route.path)
</script>

<style scoped>
.app-aside {
  background: var(--eae-surface);
  border-right: 1px solid var(--eae-border);
  display: flex;
  flex-direction: column;
}

.sidebar-brand {
  padding: var(--eae-space-4) var(--eae-space-4) var(--eae-space-3);
  border-bottom: 1px solid var(--eae-border);
}

.sidebar-brand-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--eae-text-primary);
}

.sidebar-brand-sub {
  font-size: 12px;
  color: var(--eae-text-secondary);
  margin-top: 2px;
}

.sidebar-menu {
  border-right: none;
  flex: 1;
}

.sidebar-menu .el-menu-item.is-active {
  background: var(--eae-primary-light);
  color: var(--eae-primary);
}
</style>
