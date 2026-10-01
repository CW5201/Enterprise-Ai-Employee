<template>
  <div class="page">
    <h1 class="page-title">AI 任务</h1>
    <div class="page-sub">提交企业任务，查看结果与执行过程</div>
    <div class="empty-state">
      <div class="empty-title">暂无任务</div>
      <div class="empty-desc">开始创建一个企业任务</div>
      <el-button type="primary" @click="onCreate">新建任务</el-button>
    </div>
  </div>
</template>

<script setup lang="ts">
function onCreate() {
  // Phase 6.2 fills in the real task form / navigation
}
</script>

<style scoped>
.page-sub {
  font-size: 13px;
  color: var(--eae-text-secondary);
  margin-bottom: var(--eae-space-5);
}

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
  margin: var(--eae-space-2) 0 var(--eae-space-5);
}
</style>
