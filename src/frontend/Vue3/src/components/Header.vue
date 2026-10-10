<template>
  <el-header class="app-header">
    <div class="header-inner">
      <div class="header-left">
        <el-button
          class="collapse-btn"
          text
          :icon="layout.collapsed ? Expand : Fold"
          aria-label="折叠 / 展开侧边栏"
          @click="layout.toggle()"
        />
        <span class="header-title">{{ pageTitle }}</span>
      </div>
      <div class="header-right">
        <el-dropdown trigger="click">
          <span class="header-user">
            <el-avatar :size="28">
              <el-icon><User /></el-icon>
            </el-avatar>
            <span class="header-user-name">用户</span>
          </span>
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item>
                <router-link :to="{ name: 'settings' }">系统设置</router-link>
              </el-dropdown-item>
            </el-dropdown-menu>
          </template>
        </el-dropdown>
      </div>
    </div>
  </el-header>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { Expand, Fold, User } from '@element-plus/icons-vue'
import { useLayoutStore } from '@/stores/layout'

const route = useRoute()
const layout = useLayoutStore()

const pageTitle = computed(() => (route.meta?.title as string | undefined) ?? 'Enterprise AI Employee')
</script>

<style scoped>
.app-header {
  height: var(--eae-header-height);
  background: var(--eae-surface);
  border-bottom: 1px solid var(--eae-border);
}

.header-inner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  height: 100%;
}

.header-left {
  display: flex;
  align-items: center;
  gap: var(--eae-space-2);
  min-width: 0;
}

/* optically align the icon with the 20px content gutter below */
.collapse-btn {
  margin-left: -6px;
  padding: 6px;
}

.header-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--eae-text-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.header-right {
  display: flex;
  align-items: center;
  gap: var(--eae-space-4);
}

.header-user {
  display: flex;
  align-items: center;
  gap: var(--eae-space-2);
  cursor: pointer;
  color: var(--eae-text-primary);
}

.header-user-name {
  font-size: 13px;
}
</style>
