// Settings form, pure part: the configuration's JSON schema read as sections and fields, typed
// values checked against their bounds, differences between configurations, and what they
// affect. No DOM: tested with Node (tests/js/settings_model.test.mjs).

// Where the schema keeps shared definitions (sections, enumerations).
const REF_PREFIX = '#/$defs/';
// What a change can affect, widest first (the server's `Effect`).
export const EFFECT_ORDER = Object.freeze(['labels', 'meshes', 'network', 'previews', 'none']);
// How prominently a section is shown (the server's `Level`).
export const LEVELS = Object.freeze({
  COMMON: 'common',
  ADVANCED: 'advanced',
  EXPERIMENTAL: 'experimental',
});

/**
 * @typedef {object} Field
 * @property {string} path Dotted path, e.g. `segmentation.burst.density`.
 * @property {string} name Its last part.
 * @property {string} kind `boolean`, `integer`, `number`, `choice` or `json` (lists, maps).
 * @property {Array<*>} choices Allowed values of a `choice`.
 * @property {object} bounds `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum` if set.
 * @property {string} description What it is.
 * @property {string} advice When to change it ('' if none).
 * @property {string} effect What a change affects.
 */

/**
 * @typedef {object} Section
 * @property {string} path Dotted path.
 * @property {string} description What it is about.
 * @property {string} effect What a change of its settings affects.
 * @property {string} level One of LEVELS.
 * @property {Field[]} fields Its own settings (nested sections are sections of their own).
 */

/**
 * Follow a `$ref` to its definition.
 * @param {object} schema The whole JSON schema.
 * @param {object} node A schema node.
 * @returns {object} The node, or the definition it refers to.
 */
function resolve(schema, node) {
  return node.$ref ? schema.$defs[node.$ref.slice(REF_PREFIX.length)] : node;
}

/**
 * Tell a nested section from a setting.
 * @param {object} definition A resolved schema node.
 * @returns {boolean} Whether it is a section (an object with properties).
 */
function isSection(definition) {
  return definition.type === 'object' && definition.properties !== undefined;
}

/**
 * Read the input a setting needs, and its bounds.
 * @param {object} definition The setting's resolved schema.
 * @returns {{kind: string, choices: Array<*>, bounds: object}} Its kind, choices and bounds.
 */
function kindOf(definition) {
  const bounds = {};
  for (const key of ['minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum']) {
    if (definition[key] !== undefined) bounds[key] = definition[key];
  }
  if (Array.isArray(definition.enum)) return { kind: 'choice', choices: definition.enum, bounds };
  const simple = ['boolean', 'integer', 'number'];
  const kind = simple.includes(definition.type) ? definition.type : 'json';
  return { kind, choices: [], bounds };
}

/**
 * Read the sections a form shows, depth first, in the schema's order.
 * @param {object} schema The configuration's JSON schema (`GET /api/config/schema`).
 * @param {string[]} roots Top-level sections to read, in the schema's order.
 * @param {string[]} [kept] Paths that stay the server's: left out, with what lies below them.
 * @returns {Section[]} Every section below them, each with its own settings.
 */
export function readSections(schema, roots, kept = []) {
  const sections = [];
  const isKept = (path) => kept.some((k) => path === k || path.startsWith(`${k}.`));
  const walk = (definition, path, inherited) => {
    if (isKept(path)) return;
    const section = {
      path,
      description: definition.description ?? '',
      effect: definition.effect ?? inherited,
      level: definition.level ?? LEVELS.COMMON,
      fields: [],
    };
    sections.push(section);
    for (const [name, node] of Object.entries(definition.properties)) {
      const target = resolve(schema, node);
      const fieldPath = `${path}.${name}`;
      if (isSection(target)) {
        walk(target, fieldPath, section.effect);
        continue;
      }
      if (isKept(fieldPath)) continue;
      section.fields.push({
        path: fieldPath,
        name,
        ...kindOf(target),
        description: node.description ?? target.description ?? '',
        advice: node.advice ?? '',
        effect: node.effect ?? section.effect,
      });
    }
  };
  for (const root of Object.keys(schema.properties).filter((key) => roots.includes(key))) {
    walk(resolve(schema, schema.properties[root]), root, EFFECT_ORDER[0]);
  }
  return sections;
}

/**
 * Read a value at a dotted path.
 * @param {object} config A configuration.
 * @param {string} path Dotted path.
 * @returns {*} The value (undefined if absent).
 */
export function valueAt(config, path) {
  return path.split('.').reduce((node, key) => node?.[key], config);
}

/**
 * Set a value at a dotted path (the sections on the way must exist).
 * @param {object} config A configuration, changed in place.
 * @param {string} path Dotted path.
 * @param {*} value The new value.
 */
export function setValue(config, path, value) {
  const keys = path.split('.');
  const last = keys.pop();
  keys.reduce((node, key) => node[key], config)[last] = value;
}

/**
 * Check what was typed for a setting against its kind and bounds.
 * @param {Field} field The setting.
 * @param {string|boolean} raw The input's value (a boolean for a switch).
 * @returns {{value: *}|{error: string}} The typed value, or why it is refused.
 */
export function checkValue(field, raw) {
  if (field.kind === 'boolean') return { value: Boolean(raw) };
  if (field.kind === 'choice') {
    const value = field.choices.find((choice) => JSON.stringify(choice) === raw);
    return value === undefined ? { error: 'Pick one of the choices.' } : { value };
  }
  if (field.kind === 'json') {
    try {
      return { value: JSON.parse(raw) };
    } catch {
      return { error: 'Not valid JSON (lists in [ ], text in " ").' };
    }
  }
  if (String(raw).trim() === '') return { error: 'A number is needed.' };
  const value = Number(raw);
  if (!Number.isFinite(value)) return { error: 'A number is needed.' };
  if (field.kind === 'integer' && !Number.isInteger(value)) return { error: 'A whole number is needed.' };
  const { minimum, maximum, exclusiveMinimum, exclusiveMaximum } = field.bounds;
  if (minimum !== undefined && value < minimum) return { error: `At least ${minimum}.` };
  if (exclusiveMinimum !== undefined && value <= exclusiveMinimum) {
    return { error: `More than ${exclusiveMinimum}.` };
  }
  if (maximum !== undefined && value > maximum) return { error: `At most ${maximum}.` };
  if (exclusiveMaximum !== undefined && value >= exclusiveMaximum) {
    return { error: `Less than ${exclusiveMaximum}.` };
  }
  return { value };
}

/**
 * @typedef {object} Difference
 * @property {string} path Dotted path of the setting.
 * @property {*} a Its value in the first configuration (undefined if absent).
 * @property {*} b Its value in the second.
 */

/**
 * List the settings whose values differ between two configurations (or two sections), walking
 * both; lists count as single values.
 * @param {*} a One configuration, or one section of it.
 * @param {*} b The other.
 * @param {string} [prefix] Path of what is compared (empty for a whole configuration).
 * @returns {Difference[]} One entry per differing value, in the order met.
 */
export function settingDifferences(a, b, prefix = '') {
  const differences = [];
  const isObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
  const walk = (x, y, path) => {
    if (isObject(x) && isObject(y)) {
      for (const key of new Set([...Object.keys(x), ...Object.keys(y)])) {
        walk(x[key], y[key], path ? `${path}.${key}` : key);
      }
    } else if (JSON.stringify(x) !== JSON.stringify(y)) {
      differences.push({ path, a: x, b: y });
    }
  };
  walk(a, b, prefix);
  return differences;
}

/**
 * List the settings of `used` that differ from `base`, as `section.key=value` lines (the
 * overrides a job request takes), within the sections it may override.
 * @param {object} base The server's configuration.
 * @param {object} used Another configuration.
 * @param {string[]} sections Sections a request may override.
 * @returns {string[]} One override line per differing value.
 */
export function settingsDiff(base, used, sections) {
  return sections
    .flatMap((section) => settingDifferences(base[section], used[section], section))
    .map(({ path, b }) => `${path}=${JSON.stringify(b)}`);
}

/**
 * List the settings whose value differs between two configurations.
 * @param {Field[]} fields The settings to compare.
 * @param {object} reference One configuration.
 * @param {object} current The other.
 * @returns {Field[]} Those that differ.
 */
export function changedFields(fields, reference, current) {
  const same = (field) => JSON.stringify(valueAt(reference, field.path))
    === JSON.stringify(valueAt(current, field.path));
  return fields.filter((field) => !same(field));
}

/**
 * Say what a set of changes affects: the widest effect among them.
 * @param {Field[]} changed The changed settings.
 * @param {string} reference What they were compared with, e.g. `the defaults` or a run's name.
 * @param {Object<string, string>} texts What each effect means (the schema's `effects`).
 * @returns {string} One sentence.
 */
export function describeChanges(changed, reference, texts) {
  if (changed.length === 0) return `Same settings as ${reference}.`;
  const effect = EFFECT_ORDER.find((e) => changed.some((field) => field.effect === e));
  const count = changed.length === 1 ? 'One setting differs' : `${changed.length} settings differ`;
  return `${count} from ${reference}; a change affects ${texts[effect]}.`;
}
