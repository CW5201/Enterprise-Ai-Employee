<template>
  <div class="page">
    <h1 class="page-title">报告中心</h1>
    <div class="empty-state">
      <div class="empty-title">暂无报告</div>
      <div class="empty-desc">报告生成模块尚未接入</div>
    </div>
  </div>
</template>

<script setup lang="ts"></script>

<style scoped>
.empty-state {
  padding: 64px 0;
  text-align: center;
}

.empty-title {
  font-size: 16px;
  font-weight: 600;
  color: var(--eae-text-primary);
}

.empty-desc {
  font-size: 13px;
  color: var(--eae-text-secondary);
  margin-top: var(--eae-space-2);
}
</style>
