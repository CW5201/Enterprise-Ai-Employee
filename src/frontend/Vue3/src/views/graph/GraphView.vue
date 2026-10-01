<template>
  <div class="page">
    <h1 class="page-title">知识图谱</h1>
    <div class="empty-state">
      <div class="empty-title">暂无数据</div>
      <div class="empty-desc">该模块正在建设中</div>
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
