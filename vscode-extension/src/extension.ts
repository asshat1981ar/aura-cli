import * as vscode from 'vscode';
import * as cp from 'child_process';

let outputChannel: vscode.OutputChannel;

function getAuraProcess(projectRoot: string, pythonPath: string): cp.ChildProcess {
    return cp.spawn(pythonPath, ['main.py', 'transport', '--root', projectRoot], {
        cwd: projectRoot,
        stdio: ['pipe', 'pipe', 'pipe']
    });
}

async function sendRequest(proc: cp.ChildProcess, method: string, params: object): Promise<unknown> {
    return new Promise((resolve, reject) => {
        const id = Date.now();
        const request = JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n';

        let buffer = '';
        proc.stdout?.on('data', (chunk: Buffer) => {
            buffer += chunk.toString();
            const lines = buffer.split('\n');
            buffer = lines.pop() ?? '';
            for (const line of lines) {
                if (!line.trim()) continue;
                try {
                    const msg = JSON.parse(line);
                    if (msg.id === id) {
                        if (msg.error) reject(new Error(msg.error.message));
                        else resolve(msg.result);
                    } else if (msg.method) {
                        // Notification — log to output channel
                        outputChannel.appendLine(`[${msg.method}] ${JSON.stringify(msg.params)}`);
                    }
                } catch { /* ignore parse errors */ }
            }
        });

        proc.stdin?.write(request);
        setTimeout(() => reject(new Error('AURA request timeout')), 30000);
    });
}

export function activate(context: vscode.ExtensionContext): void {
    outputChannel = vscode.window.createOutputChannel('AURA CLI');

    context.subscriptions.push(
        vscode.commands.registerCommand('aura.runGoal', async () => {
            const config = vscode.workspace.getConfiguration('aura');
            const projectRoot = config.get<string>('projectRoot') || vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || '.';
            const pythonPath = config.get<string>('pythonPath') || 'python3';

            const goal = await vscode.window.showInputBox({ prompt: 'Enter your goal for AURA', placeHolder: 'e.g. Add unit tests for auth module' });
            if (!goal) return;

            outputChannel.show();
            outputChannel.appendLine(`Running goal: ${goal}`);

            const proc = getAuraProcess(projectRoot, pythonPath);
            try {
                const result = await sendRequest(proc, 'goal', { goal, dry_run: false });
                outputChannel.appendLine(`Result: ${JSON.stringify(result)}`);
                vscode.window.showInformationMessage('AURA: Goal completed!');
            } catch (err) {
                vscode.window.showErrorMessage(`AURA Error: ${err}`);
            } finally {
                proc.kill();
            }
        }),

        vscode.commands.registerCommand('aura.status', async () => {
            const config = vscode.workspace.getConfiguration('aura');
            const projectRoot = config.get<string>('projectRoot') || '.';
            const pythonPath = config.get<string>('pythonPath') || 'python3';
            const proc = getAuraProcess(projectRoot, pythonPath);
            try {
                const result = await sendRequest(proc, 'status', {}) as { agent_count: number };
                vscode.window.showInformationMessage(`AURA: ${result.agent_count} agents active`);
            } finally {
                proc.kill();
            }
        })
    );
}

export function deactivate(): void {}
