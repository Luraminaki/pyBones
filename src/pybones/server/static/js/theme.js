// Colour theme: system (follows the OS), light or dark; remembered in this browser only.

import { recall, remember } from './ui.js';

const KEY = 'pybones.theme';
const ORDER = ['system', 'light', 'dark'];
/** Event sent on `document` when the colours change (the 3D scene repaints its background). */
export const THEME_CHANGED = 'pybones:theme-changed';

/**
 * Read the theme chosen in this browser.
 * @returns {string} `system`, `light` or `dark` (`system` if none, or storage is blocked).
 */
function stored() {
  const value = recall(KEY);
  return ORDER.includes(value) ? value : 'system';
}

/**
 * Apply a theme to the page and tell the other modules.
 * @param {string} theme `system`, `light` or `dark`.
 */
function apply(theme) {
  if (theme === 'system') delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = theme;
  document.dispatchEvent(new CustomEvent(THEME_CHANGED));
}

/**
 * Apply the remembered theme and make a button cycle through the themes.
 * @param {HTMLButtonElement} button The theme button (its text shows the theme).
 */
export function initTheme(button) {
  let theme = stored();
  apply(theme);
  button.textContent = `Theme: ${theme}`;
  button.addEventListener('click', () => {
    theme = ORDER[(ORDER.indexOf(theme) + 1) % ORDER.length];
    remember(KEY, theme);
    apply(theme);
    button.textContent = `Theme: ${theme}`;
  });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    if (theme === 'system') apply(theme);
  });
}
