<template>
  <div class="app-layout">
    <AppSidebar />
    <div class="app-main-wrap">
      <AppHeader />
      <main ref="mainRef" class="app-main">
        <router-view />
      </main>
    </div>
  </div>
</template>

<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import AppHeader from '@/components/Header.vue'
import AppSidebar from '@/components/Sidebar.vue'
import { useLayoutStore } from '@/stores/layout'

const route = useRoute()
const layout = useLayoutStore()
const mainRef = ref<HTMLElement | null>(null)

// `.app-main` — not the window — is the scroll container, so the router's
// scrollBehavior would never fire. Reset it explicitly on every navigation,
// otherwise a long page (评测 / 分析) leaves the next page scrolled down.
watch(
  () => route.fullPath,
  () =>
    nextTick(() => {
      if (mainRef.value) mainRef.value.scrollTop = 0
    }),
)

function onResize() {
  layout.syncToViewport()
}

onMounted(() => {
  layout.syncToViewport()
  window.addEventListener('resize', onResize)
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', onResize)
})
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

/* Narrow viewports: the 220px aside collapses to 64px, so the main column
   gains width, but a 20px gutter on both sides is still 40px of dead space
   on a 360px phone. Tighten it so grid panes (minmax 360px) keep room. */
@media (max-width: 640px) {
  .app-main {
    padding: var(--eae-space-3);
  }
}
</style>
