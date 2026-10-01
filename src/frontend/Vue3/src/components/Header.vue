<template>
  <div class="header-inner">
    <div class="header-left">
      <span class="header-title">Enterprise AI Employee</span>
    </div>
    <div class="header-right">
      <el-input
        v-model="search"
        placeholder="搜索"
        :prefix-icon="Search"
        class="header-search"
        clearable
      />
      <el-badge :value="0" :hidden="true">
        <el-button :icon="Bell" text />
      </el-badge>
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
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { Bell, Search, User } from '@element-plus/icons-vue'

const search = ref('')
</script>

<style scoped>
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
}

.header-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--eae-text-primary);
}

.header-right {
  display: flex;
  align-items: center;
  gap: var(--eae-space-4);
}

.header-search {
  width: 220px;
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
