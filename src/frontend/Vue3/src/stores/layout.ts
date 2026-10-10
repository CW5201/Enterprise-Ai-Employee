import { defineStore } from 'pinia'
import { ref } from 'vue'

/**
 * Shared shell state (sidebar collapse). Kept in a store because both the
 * header toggle and the aside react to it.
 *
 * Narrow viewports collapse the sidebar automatically, but only until the
 * user toggles it manually — after that the user's choice wins.
 */
export const useLayoutStore = defineStore('layout', () => {
  const collapsed = ref(false)
  const userTouched = ref(false)

  function toggle(): void {
    collapsed.value = !collapsed.value
    userTouched.value = true
  }

  function syncToViewport(): void {
    if (userTouched.value) return
    collapsed.value = window.innerWidth < 1000
  }

  return { collapsed, userTouched, toggle, syncToViewport }
})
