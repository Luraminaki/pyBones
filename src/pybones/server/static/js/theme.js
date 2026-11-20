// Colour theme: system (follows the OS), light or dark; remembered in this browser only.

const KEY = 'pybones.theme';
const ORDER = ['system', 'light', 'dark'];
export const THEME_CHANGED = 'pybones:theme-changed';

function stored() {
  try {
    const value = localStorage.getItem(KEY);
    return ORDER.includes(value) ? value : 'system';
  } catch {
    return 'system';
  }
}

function apply(theme) {
  if (theme === 'system') delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = theme;
  document.dispatchEvent(new CustomEvent(THEME_CHANGED));
}

export function initTheme(button) {
  let theme = stored();
  apply(theme);
  button.textContent = `Theme: ${theme}`;
  button.addEventListener('click', () => {
    theme = ORDER[(ORDER.indexOf(theme) + 1) % ORDER.length];
    try {
      localStorage.setItem(KEY, theme);
    } catch {
      // not remembered: still applied for this visit
    }
    apply(theme);
    button.textContent = `Theme: ${theme}`;
  });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    if (theme === 'system') apply(theme);
  });
}
