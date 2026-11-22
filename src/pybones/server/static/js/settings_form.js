// Settings form of a new run, built from the configuration's JSON schema: every setting a run
// may change, with what it is, when to change it and what a change affects. Values are checked
// as typed, changes are marked, and the server's own checks come back next to their setting.

import { el } from './ui.js';
import {
  changedFields, checkValue, describeChanges, LEVELS, readSections, setValue, settingsDiff, valueAt,
} from './settings_model.js';

// Groups of sections, in the order shown, with their heading.
const GROUPS = [
  [LEVELS.EXPERIMENTAL, 'Experimental (off by default)'],
  [LEVELS.COMMON, 'Settings'],
  [LEVELS.ADVANCED, 'Advanced'],
];
// First element of an invalid setting's location in the server's answer (then its path).
const CONFIG_LOCATION = 'config';
const DEFAULTS = 'the defaults';

/**
 * @typedef {object} SettingsForm
 * @property {function(): string[]} overrides The changed settings, as override lines.
 * @property {function(): boolean} valid Whether every typed value is acceptable.
 * @property {function(object, string): void} startFrom Fill in a run's settings.
 * @property {function(*): boolean} showErrors Show the server's errors next to their settings.
 * @property {function(): void} open Open the form.
 */

/**
 * Write a value into a setting's input.
 * @param {import('./settings_model.js').Field} field The setting.
 * @param {HTMLInputElement|HTMLSelectElement} input Its input.
 * @param {*} value The value.
 */
function display(field, input, value) {
  if (field.kind === 'boolean') input.checked = Boolean(value);
  else if (field.kind === 'choice' || field.kind === 'json') input.value = JSON.stringify(value);
  else input.value = String(value);
}

/**
 * Build the input a setting needs.
 * @param {import('./settings_model.js').Field} field The setting.
 * @param {string} id The input's id.
 * @param {function(): void} onChange Called after each change.
 * @returns {HTMLInputElement|HTMLSelectElement} The input.
 */
function inputFor(field, id, onChange) {
  const described = `${id}-help`;
  if (field.kind === 'boolean') {
    return el('input', { type: 'checkbox', id, 'aria-describedby': described, onchange: onChange });
  }
  if (field.kind === 'choice') {
    const options = field.choices.map((c) => el('option', { value: JSON.stringify(c) }, String(c)));
    return el('select', { id, 'aria-describedby': described, onchange: onChange }, ...options);
  }
  if (field.kind === 'json') {
    return el('input', {
      type: 'text', id, spellcheck: 'false', 'aria-describedby': described, onchange: onChange,
    });
  }
  const { minimum, maximum } = field.bounds;
  return el('input', {
    type: 'number', id, step: field.kind === 'integer' ? '1' : 'any', min: minimum, max: maximum,
    'aria-describedby': described, onchange: onChange,
  });
}

/**
 * Build the settings form into its dialog.
 * @param {object} schema The configuration's JSON schema.
 * @param {object} base The server's configuration (the defaults of a new run).
 * @param {string[]} roots Sections a run may change.
 * @returns {SettingsForm} The form.
 */
export function initSettingsForm(schema, base, roots) {
  const dialog = document.getElementById('settings-dialog');
  const body = document.getElementById('settings-body');
  const summaries = [document.getElementById('settings-summary'), document.getElementById('settings-effect')];
  const advanced = document.getElementById('settings-advanced');
  const texts = schema.effects;
  const sections = readSections(schema, roots);
  const fields = sections.flatMap((s) => s.fields);
  const rows = new Map(); // setting path -> {row, input, error}
  const counters = new Map(); // section path -> its count of changes
  const invalid = new Set();
  let current = structuredClone(base);
  let reference = null; // {name, config} of the run the form started from

  const refresh = () => {
    for (const field of fields) {
      const changed = JSON.stringify(valueAt(current, field.path)) !== JSON.stringify(valueAt(base, field.path));
      rows.get(field.path).row.classList.toggle('changed', changed);
    }
    for (const section of sections) {
      const count = changedFields(section.fields, base, current).length;
      const counter = counters.get(section.path);
      if (counter) counter.textContent = count ? ` · ${count} changed` : '';
    }
    const against = reference ?? { name: DEFAULTS, config: base };
    const text = describeChanges(changedFields(fields, against.config, current), against.name, texts);
    for (const node of summaries) node.textContent = text;
  };

  const showError = (field, message) => {
    const { row, error } = rows.get(field.path);
    error.textContent = message;
    error.hidden = !message;
    row.classList.toggle('invalid', Boolean(message));
    if (message) {
      invalid.add(field.path);
      row.closest('details').open = true;
      if (row.closest('[data-level]').dataset.level === LEVELS.ADVANCED) advanced.checked = true;
      advanced.dispatchEvent(new Event('change'));
    } else {
      invalid.delete(field.path);
    }
  };

  const fieldRow = (field, section) => {
    const id = `setting-${field.path}`;
    const input = inputFor(field, id, () => {
      const result = checkValue(field, field.kind === 'boolean' ? input.checked : input.value);
      if ('error' in result) {
        showError(field, result.error);
        return;
      }
      showError(field, '');
      setValue(current, field.path, result.value);
      refresh();
    });
    const reset = () => {
      setValue(current, field.path, structuredClone(valueAt(base, field.path)));
      display(field, input, valueAt(current, field.path));
      showError(field, '');
      refresh();
    };
    const own = field.effect !== section.effect ? ` A change affects ${texts[field.effect]}.` : '';
    const help = el('p', { class: 'hint', id: `${id}-help` }, field.description,
      field.advice && el('span', { class: 'advice' }, ` When to change it: ${field.advice}`), own);
    const error = el('p', { class: 'setting-error', role: 'alert', hidden: true });
    const row = el('div', { class: 'setting' },
      el('label', { for: id }, el('code', {}, field.name), el('span', { class: 'changed-mark' }, 'changed')),
      el('div', { class: 'setting-input' }, input,
        el('button', { type: 'button', class: 'ghost small', title: 'Back to the default', onclick: reset }, 'Reset')),
      help, error);
    rows.set(field.path, { row, input, error });
    return row;
  };

  for (const [level, heading] of GROUPS) {
    const members = sections.filter((s) => s.level === level && s.fields.length);
    if (!members.length) continue;
    const group = el('section', { class: 'settings-group', dataset: { level } }, el('h3', {}, heading));
    for (const section of members) {
      const counter = el('span', { class: 'section-changes' });
      counters.set(section.path, counter);
      group.append(el('details', { class: 'settings-section', open: level === LEVELS.EXPERIMENTAL },
        el('summary', {}, el('code', {}, section.path), ` ${section.description}`, counter),
        el('p', { class: 'hint section-effect' }, `A change here affects ${texts[section.effect]}.`),
        ...section.fields.map((field) => fieldRow(field, section))));
    }
    body.append(group);
  }
  const advancedGroup = body.querySelector(`[data-level="${LEVELS.ADVANCED}"]`);
  advanced.addEventListener('change', () => { if (advancedGroup) advancedGroup.hidden = !advanced.checked; });
  advanced.dispatchEvent(new Event('change'));

  const fill = () => {
    for (const field of fields) display(field, rows.get(field.path).input, valueAt(current, field.path));
    for (const field of fields) showError(field, '');
    refresh();
  };
  document.getElementById('settings-reset').addEventListener('click', () => {
    current = structuredClone(base);
    reference = null;
    fill();
  });
  document.getElementById('settings-open').addEventListener('click', () => dialog.showModal());
  fill();

  return {
    overrides: () => settingsDiff(base, current, roots),
    valid: () => invalid.size === 0,
    startFrom: (config, name) => {
      current = structuredClone(base);
      for (const root of roots) current[root] = structuredClone(config[root]);
      reference = { name, config };
      fill();
    },
    showErrors: (detail) => {
      if (!Array.isArray(detail)) return false;
      let located = 0;
      for (const { loc = [], msg } of detail) {
        if (loc[0] !== CONFIG_LOCATION) continue;
        const parts = loc.slice(1).map(String);
        for (let k = parts.length; k > 0; k -= 1) {
          const field = fields.find((f) => f.path === parts.slice(0, k).join('.'));
          if (field) {
            showError(field, msg);
            located += 1;
            break;
          }
        }
      }
      return located > 0;
    },
    open: () => dialog.showModal(),
  };
}
