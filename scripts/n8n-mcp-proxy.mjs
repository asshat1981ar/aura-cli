#!/usr/bin/env node
/**
 * n8n MCP Stdio Proxy
 * Bridges n8n's Streamable HTTP MCP transport to stdio for Claude Code.
 *
 * initialize and tools/list respond instantly (no network).
 * tools/call forwards directly to n8n (stateless — no session needed).
 */

const N8N_URL = process.env.N8N_MCP_URL || 'http://localhost:5678/mcp-server/http';
const N8N_TOKEN = process.env.N8N_MCP_TOKEN || '';

const baseHeaders = {
  'Content-Type': 'application/json',
  'Accept': 'application/json, text/event-stream',
};
if (N8N_TOKEN) {
  baseHeaders['Authorization'] = `Bearer ${N8N_TOKEN}`;
}

import { appendFileSync } from 'fs';
const DBG = '/tmp/n8n-proxy-live.log';
const log = (m) => { try { appendFileSync(DBG, `${Date.now()} ${m}\n`); } catch {} };

process.on('uncaughtException', (e) => log(`UNCAUGHT: ${e.stack}`));
process.on('unhandledRejection', (e) => log(`REJECT: ${e?.stack || e}`));

function isNotification(msg) {
  return (msg.id === undefined || msg.id === null) && msg.method;
}

const TOOLS = [
  { name: 'search_workflows', description: 'Search for workflows with optional filters. Returns a preview of each workflow.',
    inputSchema: { type: 'object', properties: { limit: { type: 'integer', description: 'Limit results (max 200)' }, query: { type: 'string', description: 'Filter by name or description' }, projectId: { type: 'string' } }, additionalProperties: false } },
  { name: 'execute_workflow', description: 'Execute a workflow by ID. Returns execution ID and status. Use get_execution to get full results. Use get_workflow_details first to check input schema.',
    inputSchema: { type: 'object', properties: { workflowId: { type: 'string', description: 'The ID of the workflow to execute' }, executionMode: { type: 'string', enum: ['manual', 'production'], description: 'Use "manual" for draft, "production" for active version' }, inputs: { type: 'object', description: 'Inputs for the workflow (chat, form, or webhook type)' } }, required: ['workflowId'], additionalProperties: false } },
  { name: 'get_execution', description: 'Get full execution details and results using execution ID and workflow ID.',
    inputSchema: { type: 'object', properties: { workflowId: { type: 'string', description: 'Workflow ID' }, executionId: { type: 'string', description: 'Execution ID' } }, required: ['workflowId', 'executionId'], additionalProperties: false } },
  { name: 'get_workflow_details', description: 'Get detailed information about a specific workflow including trigger details.',
    inputSchema: { type: 'object', properties: { workflowId: { type: 'string', description: 'Workflow ID' } }, required: ['workflowId'], additionalProperties: false } },
  { name: 'publish_workflow', description: 'Publish (activate) a workflow for production execution.',
    inputSchema: { type: 'object', properties: { workflowId: { type: 'string', description: 'Workflow ID' }, versionId: { type: 'string', description: 'Optional version ID' } }, required: ['workflowId'], additionalProperties: false } },
  { name: 'unpublish_workflow', description: 'Unpublish (deactivate) a workflow.',
    inputSchema: { type: 'object', properties: { workflowId: { type: 'string', description: 'Workflow ID' } }, required: ['workflowId'], additionalProperties: false } },
  { name: 'search_nodes', description: 'Search for n8n nodes by service name, trigger type, or utility function.',
    inputSchema: { type: 'object', properties: { queries: { type: 'array', items: { type: 'string' }, description: 'Search queries for n8n nodes' } }, required: ['queries'], additionalProperties: false } },
  { name: 'get_node_types', description: 'Get TypeScript type definitions for n8n nodes. MUST be called before writing workflow code.',
    inputSchema: { type: 'object', properties: { nodeIds: { type: 'array', items: { type: 'string' }, description: 'Node IDs (strings or objects with discriminators)' } }, required: ['nodeIds'], additionalProperties: false } },
  { name: 'get_suggested_nodes', description: 'Get curated node recommendations for workflow technique categories.',
    inputSchema: { type: 'object', properties: { categories: { type: 'array', items: { type: 'string' }, description: 'Categories: chatbot, notification, scheduling, data_transformation, data_persistence, data_extraction, document_processing, form_input, content_generation, triage, scraping_and_research' } }, required: ['categories'], additionalProperties: false } },
  { name: 'validate_workflow', description: 'Validate n8n Workflow SDK code. Always validate before creating a workflow.',
    inputSchema: { type: 'object', properties: { code: { type: 'string', description: 'Full workflow SDK code' } }, required: ['code'], additionalProperties: false } },
  { name: 'create_workflow_from_code', description: 'Create a workflow in n8n from validated SDK code.',
    inputSchema: { type: 'object', properties: { code: { type: 'string', description: 'Validated workflow SDK code' }, name: { type: 'string', description: 'Workflow name (max 128 chars)' }, description: { type: 'string', description: 'Short description (max 255 chars)' }, projectId: { type: 'string', description: 'Optional project ID' } }, required: ['code'], additionalProperties: false } },
  { name: 'archive_workflow', description: 'Archive a workflow in n8n by its ID.',
    inputSchema: { type: 'object', properties: { workflowId: { type: 'string', description: 'Workflow ID' } }, required: ['workflowId'], additionalProperties: false } },
  { name: 'update_workflow', description: 'Update an existing workflow from validated SDK code.',
    inputSchema: { type: 'object', properties: { workflowId: { type: 'string', description: 'Workflow ID' }, code: { type: 'string', description: 'Validated workflow SDK code' }, name: { type: 'string', description: 'Workflow name' }, description: { type: 'string', description: 'Short description' } }, required: ['workflowId', 'code'], additionalProperties: false } },
];

function respond(id, result) {
  process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id, result }) + '\n');
}

function respondError(id, code, message) {
  process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id, error: { code, message } }) + '\n');
}

async function sendToN8n(message) {
  const res = await fetch(N8N_URL, {
    method: 'POST',
    headers: baseHeaders,
    body: JSON.stringify(message),
  });

  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`n8n returned ${res.status}: ${body.slice(0, 200)}`);
  }

  const ct = res.headers.get('content-type') || '';
  if (ct.includes('text/event-stream')) {
    const text = await res.text();
    const results = [];
    for (const line of text.split('\n')) {
      if (line.startsWith('data: ')) {
        try { results.push(JSON.parse(line.slice(6))); } catch {}
      }
    }
    if (results.length === 0) throw new Error('Empty SSE response from n8n');
    return results;
  }
  const text = await res.text();
  if (!text.trim()) throw new Error('Empty response from n8n');
  return [JSON.parse(text)];
}

async function handleMessage(message) {
  if (isNotification(message)) return;

  // Instant responses — no network
  if (message.method === 'initialize') {
    return respond(message.id, {
      protocolVersion: message.params?.protocolVersion || '2024-11-05',
      capabilities: { tools: {} },
      serverInfo: { name: 'n8n-mcp-proxy', version: '1.0.0' },
    });
  }

  if (message.method === 'tools/list') {
    return respond(message.id, { tools: TOOLS });
  }

  // tools/call → forward directly to n8n (stateless, no session needed)
  log(`PROXY: ${message.method} name=${message.params?.name} id=${message.id}`);
  try {
    const responses = await sendToN8n(message);
    for (const resp of responses) {
      resp.id = message.id;
      const out = JSON.stringify(resp);
      log(`RESPONSE: id=${message.id} len=${out.length}`);
      process.stdout.write(out + '\n');
    }
  } catch (err) {
    log(`ERROR: id=${message.id} ${err.message}`);
    respondError(message.id, -32000, err.message);
  }
}

let buffer = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => {
  buffer += chunk;
  let idx;
  while ((idx = buffer.indexOf('\n')) !== -1) {
    const line = buffer.slice(0, idx).trim();
    buffer = buffer.slice(idx + 1);
    if (!line) continue;
    try { handleMessage(JSON.parse(line)); } catch {}
  }
});
process.stdin.on('end', () => setTimeout(() => process.exit(0), 2000));
process.stdin.resume();
