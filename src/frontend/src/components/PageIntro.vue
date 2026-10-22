<script setup lang="ts">
defineProps<{ eyebrow: string; title: string; description: string }>();
</script>

<template>
  <section class="page-intro">
    <div class="intro-eyebrow">{{ eyebrow }}</div>
    <h1>{{ title }}</h1>
    <p>{{ description }}</p>
  </section>
</template>

<style scoped>
.page-intro { margin-bottom: 26px; }
.intro-eyebrow { color: var(--accent); font-size: 11px; font-weight: 700; letter-spacing: .12em; margin-bottom: 8px; }
h1 { margin: 0 0 8px; font-size: clamp(25px,2.5vw,32px); font-weight: 650; letter-spacing: -.7px; }
p { margin: 0; color: var(--muted); font-size: 13px; }
</style>
