/**
 * Locate the Codex presentations runtime used by the .mjs deck builders here.
 *
 * The builders used to import the runtime helpers from a hardcoded
 * `/Users/<name>/.codex/...` path. This resolves the same locations from the
 * environment instead, reading the gitignored repo-root `.env` the way
 * aethmodular_cli/env.py does for Python (a variable already set wins; `~` is
 * expanded here). See `.env.example`:
 *
 *   AETHMODULAR_CODEX_PRESENTATIONS_DIR  the presentations skill directory
 *   AETHMODULAR_CODEX_PYTHON             the runtime's python3, for finalization
 *
 * Defaults are the paths the builders were written against, under
 * $CODEX_HOME (else ~/.codex) and ~/.cache.
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {parseEnv} from 'node:util';

// workflows -> ftir_hips_chem -> research -> repo
const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const ENV_FILE = path.join(REPO_ROOT, '.env');

if (fs.existsSync(ENV_FILE)) {
  for (const [key, value] of Object.entries(parseEnv(fs.readFileSync(ENV_FILE, 'utf8')))) {
    if (!(key in process.env) && value) process.env[key] = value;
  }
}

const expand = p => (p === '~' || p.startsWith('~/') ? path.join(os.homedir(), p.slice(1)) : p);
const fromEnv = (name, fallback) => expand(process.env[name] || fallback);

const codexHome = fromEnv('CODEX_HOME', path.join(os.homedir(), '.codex'));

/** The presentations skill directory (holds container_tools/). */
export const skill = fromEnv(
  'AETHMODULAR_CODEX_PRESENTATIONS_DIR',
  path.join(codexHome, 'plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations'),
);

/** python3 from the Codex runtime, passed to finalizePresentation(). */
export const pythonExecutable = fromEnv(
  'AETHMODULAR_CODEX_PYTHON',
  path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'),
);

const containerTool = name => import(pathToFileURL(path.join(skill, 'container_tools', name)).href);

/** Exports of container_tools/runtime_helpers.mjs (importRuntimeModule, ...). */
export const runtimeHelpers = () => containerTool('runtime_helpers.mjs');

/** Exports of container_tools/artifact_tool_utils.mjs (finalizePresentation, ...). */
export const artifactToolUtils = () => containerTool('artifact_tool_utils.mjs');
