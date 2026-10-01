<template>
  <el-container class="app-layout">
    <AppSidebar />
    <el-container class="app-main-wrap">
      <AppHeader />
      <el-main class="app-main">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup lang="ts">
import AppHeader from '@/components/Header.vue'
import AppSidebar from '@/components/Sidebar.vue'
</script>

<style scoped>
.app-layout {
  height: 100%;
}

.app-main-wrap {
  flex-direction: column;
  min-width: 0;
}

.app-main {
  padding: var(--eae-space-5);
  overflow-y: auto;
  background: var(--eae-bg);
}
</style>
