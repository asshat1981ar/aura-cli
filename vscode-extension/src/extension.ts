import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {
    context.subscriptions.push(
        vscode.commands.registerCommand('aura.runGoal', () => {
            vscode.window.showInputBox({ prompt: 'Enter goal for AURA' });
        }),
        vscode.commands.registerCommand('aura.status', () => {
            vscode.window.showInformationMessage('AURA: checking status...');
        })
    );
}

export function deactivate() {}
