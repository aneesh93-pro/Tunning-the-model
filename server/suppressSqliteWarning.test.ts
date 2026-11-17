import test from 'node:test';
import assert from 'node:assert/strict';
import { isSqliteExperimentalWarning } from './suppressSqliteWarning.ts';

test('identifies node:sqlite ExperimentalWarning strings and Error objects', () => {
  assert.equal(
    isSqliteExperimentalWarning(
      'SQLite is an experimental feature and might change at any time',
      'ExperimentalWarning',
    ),
    true,
  );
  assert.equal(
    isSqliteExperimentalWarning(
      Object.assign(new Error('SQLite is an experimental feature and might change at any time'), {
        name: 'ExperimentalWarning',
      }),
    ),
    true,
  );
});

test('does not match unrelated warnings or non-ExperimentalWarning types', () => {
  assert.equal(
    isSqliteExperimentalWarning('DeprecationWarning: something else', 'DeprecationWarning'),
    false,
  );
  assert.equal(
    isSqliteExperimentalWarning('SQLite is an experimental feature', 'DeprecationWarning'),
    false,
  );
  assert.equal(
    isSqliteExperimentalWarning('Fetch API is an experimental feature', 'ExperimentalWarning'),
    false,
  );
});
