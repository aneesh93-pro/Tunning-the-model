export function isSqliteExperimentalWarning(
  warning: string | Error | { name?: string; message?: string },
  type?: string,
): boolean {
  const message = typeof warning === 'string' ? warning : warning?.message ?? '';
  const warningType =
    type ?? (typeof warning === 'object' && warning !== null ? warning.name : undefined);

  if (warningType && warningType !== 'ExperimentalWarning') {
    return false;
  }

  return message.includes('SQLite is an experimental feature');
}

const FLAG = Symbol.for('minimind.sqliteWarningFilterInstalled');

export function installSqliteWarningFilter(targetProcess: NodeJS.Process = process): void {
  const proc = targetProcess as NodeJS.Process & Record<symbol, boolean | undefined>;
  if (proc[FLAG]) return;
  proc[FLAG] = true;

  const originalEmitWarning = targetProcess.emitWarning.bind(targetProcess);
  targetProcess.emitWarning = ((warning: string | Error, ...args: unknown[]) => {
    const typeArg = typeof args[0] === 'string' ? args[0] : undefined;
    if (isSqliteExperimentalWarning(warning, typeArg)) {
      return;
    }
    return (originalEmitWarning as (...params: unknown[]) => void)(warning, ...args);
  }) as NodeJS.Process['emitWarning'];
}

installSqliteWarningFilter();
