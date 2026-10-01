<template>
  <div class="app-layout">
    <AppSidebar />
    <div class="app-main-wrap">
      <AppHeader />
      <main class="app-main">
        <router-view />
      </main>
    </div>
  </div>
</template>

<script setup lang="ts">
import AppHeader from '@/components/Header.vue'
import AppSidebar from '@/components/Sidebar.vue'
</script>

<style scoped>
.app-layout {
  display: flex;
  height: 100%;
}

.app-main-wrap {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.app-main {
  flex: 1;
  overflow-y: auto;
  padding: var(--eae-space-5);
  background: var(--eae-bg);
}
</style>
