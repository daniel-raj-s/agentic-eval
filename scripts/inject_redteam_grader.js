#!/usr/bin/env node
/**
 * Add a fully configured Groq grader to a generated Promptfoo red-team config.
 *
 * `promptfoo redteam generate` drops `defaultTest.options` from the config it
 * writes, and the CLI `--grader` flag only accepts a provider string, which
 * cannot set `reasoning_format: hidden`. Without that setting Groq's
 * gpt-oss models can return empty content and the red-team grader stalls.
 */
const fs = require('fs');
const path = require('path');

const target = process.argv[2] || 'tests/promptfoo/redteam-generated.yaml';
const file = path.resolve(target);
const text = fs.readFileSync(file, 'utf8');
const nl = text.includes('\r\n') ? '\r\n' : '\n';
const lines = text.split(/\r?\n/);

const index = lines.indexOf('defaultTest:');
if (index === -1) {
  console.error(`No top-level defaultTest block found in ${target}`);
  process.exit(1);
}

const block = [
  '  options:',
  '    provider:',
  '      id: groq:openai/gpt-oss-20b',
  '      config:',
  '        reasoning_format: hidden',
];

if (lines[index + 1] === '  options:') {
  let end = index + 2;
  while (end < lines.length && (lines[end].startsWith('    ') || lines[end].trim() === '')) {
    end += 1;
  }
  lines.splice(index + 1, end - index - 1, ...block);
  console.log(`Updated Groq grader config in ${target}`);
} else {
  lines.splice(index + 1, 0, ...block);
  console.log(`Injected Groq grader config into ${target}`);
}

fs.writeFileSync(file, lines.join(nl), 'utf8');
